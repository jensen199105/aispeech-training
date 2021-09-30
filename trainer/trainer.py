"""The core of trainer object"""
import logging
import warnings

import torch
import torch.distributed as dist
from apex import amp  # pylint: disable=import-error

from .common import Metric, Timer
from .optimizer import DistributedOptimizer, EarlyStop
from ..utils.common import reduce_number

logger = logging.getLogger(__name__)


class TrainerLogAdapter(logging.LoggerAdapter):
    """Add contextual information to logger

    The training mode, epoch
    """

    def process(self, msg, kwargs):
        epoch_mode = 'TR' if self.extra['is_training'] else 'CV'
        epoch = self.extra['epoch']
        meta_log = f'[{epoch_mode} epoch {epoch}] '
        return meta_log + msg, kwargs


class Trainer():
    """Model trainer

    Train a model by iterating the data_queue and optimizing the Loss function.

    Arguments:
        model:
        loss:
        optimizer:
        bmuf_optimizer:
        lr_scheduler:
        checkpoint:
        log_interval:
        dump_interval: dump intermediate checkpoint every other dump_interval batches
        merge_size:
        rank:
        amp_opt_level (str, {'O1', 'O2', 'O3'}): the fp16 optimization level, see `apex.amp`
    """

    def __init__(self, model, loss, optimizer, bmuf_optimizer, lr_scheduler, checkpoint,
                 log_interval, merge_size, rank=0, amp_opt_level='O0', dump_interval=10000):
        model, optimizer = amp.initialize(model, optimizer, loss_scale=1.0, opt_level=amp_opt_level)
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.bmuf_optimizer = bmuf_optimizer
        self.lr_scheduler = lr_scheduler
        self.checkpoint = checkpoint
        self.log_interval = log_interval
        self.merge_size = merge_size
        self.rank = rank
        self.amp_opt_level = amp_opt_level
        self.dump_interval = dump_interval
        self.dist_optim = DistributedOptimizer(
            self.optimizer,
            self.bmuf_optimizer,
            self.merge_size,
        )
        self.timer = Timer()

    def train(self, data_queue, epoch):
        """Run a training epoch until the data_queue gives a None batch"""
        return self._one_epoch(data_queue, epoch, is_training=True)

    def validate(self, data_queue, epoch):
        """Run a validation epoch until the data_queue gives a None batch"""
        with torch.no_grad():
            return self._one_epoch(data_queue, epoch, is_training=False)

    def set_training_mode(self, mode):
        """Set the training mode for parts in trainer

        Args:
            mode (boolean): the training mode
        """
        # equals to module.eval if mode is False
        self.model.train(mode)
        self.loss.train(mode)

    def _log_progress(self, epoch_logger, local_metric, epoch_metric, time_result):
        """Log the epoch progress every self.log_interval batches"""

        total_batch = epoch_metric['count']
        throughput = local_metric['total_frames'] / time_result['wall']
        data_hour = epoch_metric['total_frames'] / 100 / 3600
        progress_log = f'batch {total_batch:.0f} ({data_hour:.1f}h)'
        profile_log = f'{time_result:.1f} fps {throughput:.1f}'
        epoch_logger.info('{} | {} | {}'.format(
            progress_log, self.loss.log_line(local_metric), profile_log,
        ))

    def _forward_batch(self, unused_batch_idx, batch):
        """Forward propagation of the model

        It feeds the batch to model and loss, returns the loss value and loss_stat
        """
        with self.timer['gpu']:
            output = self.model(batch)
        with self.timer['loss']:
            loss, loss_statistics = self.loss(output, batch)
        return loss, loss_statistics

    def _backward_batch(self, loss):
        """Backward propagation of the model

        It does gradient computation and post-processing
        """
        with self.timer['gpu']:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            if self.amp_opt_level == 'O2':
                warnings.warn('Customized grad_post_processing is not supported in amp_opt_level=O2. '
                              'Clipping L2-norm of gradients of the whole model into 5000.')
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 5000)
            else:
                self.model.grad_post_processing()

    def _one_epoch(self, data_queue, epoch, is_training):
        self.set_training_mode(is_training)
        # Synchronize the model parameters before each epoch
        for param in self.model.parameters():
            dist.broadcast(param, src=0)

        # Profiling timers
        self.timer.clear()
        self.timer['wall'].start()

        # Accumulate `err_label`, `labels`, `loss`
        epoch_metric = Metric()
        local_metric = Metric()

        context_info = {
            'epoch': epoch,
            'is_training': is_training,
        }
        epoch_logger = TrainerLogAdapter(logger, context_info)

        # Model merge is only required in training mode
        if is_training:
            self.dist_optim.clear()

        batch_idx = -1
        for batch_idx, batch in enumerate(self.timer['io'].profile(data_queue)):
            assert isinstance(batch, dict)

            if is_training and batch_idx % self.dump_interval == 0 and self.rank == 0:
                self.checkpoint.save_dump()

            loss, loss_statistics = self._forward_batch(batch_idx, batch)
            frames = loss_statistics['total_frames']
            if is_training:
                self._backward_batch(loss)
                # The learning rate scheduler.step MUST be called before optimizer step for cold start
                self.lr_scheduler.step()
                with self.timer['merge']:
                    try:
                        self.dist_optim.step(frames)
                    except EarlyStop:
                        data_queue.clear()
                        break

            # the last interval will be retained
            self.timer['wall'].checkpoint()
            self.timer['wall'].start()

            epoch_metric.accumulate(loss_statistics)
            local_metric.accumulate(loss_statistics)

            if local_metric['total_frames'] > self.log_interval:
                self._log_progress(epoch_logger, local_metric, epoch_metric, self.timer.last_result)
                local_metric.clear()
        else:
            # Runs only when data batch consumed (normal exit)
            epoch_logger.debug('Batches clear, epoch end')
            stop_flag = 1
            reduce_number(stop_flag)

        if batch_idx == -1:
            raise RuntimeError('Data queue is empty')

        # An extra BM step to synchronize remaining gradient updates
        # NOTE: Do this step only in training phase
        with self.timer['merge']:
            if is_training:
                self.dist_optim.force_step()

        self.timer['wall'].checkpoint()
        epoch_logger.info('----------Local epoch finished---------')
        self._log_progress(epoch_logger, epoch_metric, epoch_metric, self.timer.total_result)

        # synchronize the epoch status among all workers
        reduced_epoch_metric = epoch_metric.all_reduce()

        epoch_logger.info('----------All worker finished---------')
        total_batches = reduced_epoch_metric['count']
        total_frames = reduced_epoch_metric['total_frames']
        avg_loss = reduced_epoch_metric['loss'] / total_frames
        if self.rank == 0:
            epoch_logger.warning('{:.0f} batches, {:.0f} frames, {}'.format(
                total_batches, total_frames, self.loss.log_line(reduced_epoch_metric),
            ))

        return avg_loss
