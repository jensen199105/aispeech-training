"""THE ONLY entry point of the pytorch-asr framework"""
import logging
from pathlib import Path

import torch.optim as optim

from ..utils.split import get_per_worker_rspecs
from ..utils.auto_rank import auto_rank
from ..utils.bootstrap import init_distributed, determinize_random_state, setup_root_logger
from ..trainer.checkpoint import Checkpoint
from ..trainer.lr_scheduler import decision
from ..trainer.optimizer import BMOptimizer
from ..trainer.trainer import Trainer
from ..data.batch_loader import BatchLoader
from ..model import build_model
from ..loss import build_loss
from ..data.collector import build_collector
from ..data.utterance_reader import UtteranceReader
from ..trainer.lr_scheduler import build_scheduler
import pdb

# Don't worry to get the logger before setting up root logger
# In python3 the sub-loggers are changed by default when root
# logger is changed.
logger = logging.getLogger(__name__)


def launch(args):
    """Launch a single worker instance for training

    Args:
        - args: the arg from parse_args
    """
    ckpt_dir = Path(args.checkpoint_dir)
    data_dir = Path(args.data_dir)

    rank = auto_rank(args.world_size, filename=ckpt_dir / 'auto_rank')

    setup_root_logger(ckpt_dir, rank, args.log_debug)
    init_distributed(args.backend, args.world_size, rank, ckpt_dir)
    determinize_random_state()
    logger.warning(args)

    # Setup TR&CV dataset parameters (scp, skip_frame etc.)
    all_data = [args.feat, args.ali, args.ivec, args.lat]
    if args.inplace_split:
        sdata_dir = data_dir
    else:
        sdata_dir = ckpt_dir / 'data'
    data = {}
    for stage in args.stage:  # args.stage can be any combination of {'tr', 'cv'}
        data_rspec = get_per_worker_rspecs(
            all_data, data_dir=data_dir, sdata_dir=sdata_dir,
            stage=stage, rank=rank, world_size=args.world_size, no_split=args.no_split,
        )
        utterance_reader = UtteranceReader(data_rspec, args.random_sweep, args.use_sds)
        data[stage] = BatchLoader(utterance_reader, build_collector(args.collector))
    print('is about to build model')
    model = build_model(args.model)
    loss = build_loss(args.loss)

    # Local optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2_factor, nesterov=args.nesterov)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09)

    # Distributed training optimizer
    bmuf_optimizer = BMOptimizer(model.parameters(), method=args.merge_function, momentum=args.global_momentum)
    # Scheduler needs optimzer as arguments, but we cannot pass it in command line interface.
    # So we manually insert the optimizer object into scheduler args
    args.scheduler['optimizer'] = optimizer
    lr_scheduler = build_scheduler(args.scheduler)
    all_states = {
        'model': model,
        'lr_scheduler': lr_scheduler,
        'optimizer': optimizer,
        'bmuf_optimizer': bmuf_optimizer,
        'loss': loss,
    }
    checkpoint = Checkpoint(
        args.model, all_states, root_dir=ckpt_dir, max_keep=args.max_keep,
    )

    # args.resume, args.init and args.kaldi_init are all mutual exclusive
    if args.resume:
        checkpoint.load_best()
    elif args.init:
        checkpoint.init_model(args.init)
    elif args.kaldi_init:
        checkpoint.init_model(args.kaldi_init, from_kaldi=True)

    model.cuda()
    logger.info(data)
    logger.info(model)
    logger.info(loss)
    logger.info(optimizer)
    logger.info(bmuf_optimizer)
    logger.info(lr_scheduler)

    # Trainer holds all the essentials to train the model given the data batches
    trainer = Trainer(model, loss, optimizer, bmuf_optimizer, lr_scheduler, checkpoint,
                      args.log_interval, args.merge_size, rank,
                      amp_opt_level=args.amp_opt_level, dump_interval=args.dump_interval)

    # Training
    for epoch in range(checkpoint.epoch + 1, args.max_epoch + 1):
        if 'tr' in args.stage:
            trainer.train(data['tr'], epoch)

        # if CV is not in args.stage, pass a constant CV loss value to lr_scheduler
        if 'cv' in args.stage:
            cv_loss = trainer.validate(data['cv'], epoch)
        else:
            cv_loss = 1

        if 'tr' in args.stage:
            decisions = lr_scheduler.step_epoch(cv_loss)

            if decision.STOP in decisions:
                # remove dump file when end of training
                checkpoint.remove_dump()
                break

            # avoid doing judgements duplicated
            if rank == 0:
                if decision.ACCEPT in decisions:
                    checkpoint.save(epoch)

                # avoid loading model from dump file, remove the file if REJECT
                if decision.REJECT in decisions:
                    checkpoint.remove_dump()
                    # Scheduler should keep the information of **failed** epoch
                    checkpoint.load_best(load_scheduler=False)

    logger.warning('End of Training')
