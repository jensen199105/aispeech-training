import logging
from collections import deque
from pathlib import Path

import torch

from ..model import build_model
from ..utils.common import rm_file
from ..utils import transform_kaldi_nnet as trans_kaldi

logger = logging.getLogger(__name__)


class Checkpoint:
    """Helper class to persistent current state to storage

    Args:
        hparams (dict): the hyperparameters of the model
        target_dict (dict): the state dict to save onto filesystem
        root_dir (Path or str): checkpoint root dir
        max_keep (int): maximum number of checkpoints to keep
    """
    def __init__(self, hparams, target_dict, root_dir, max_keep=2, dump_suffix='dump'):
        if max_keep < 2:
            raise ValueError('max_keep must be greater than 2')
        self.hparams = hparams
        self.root_path = Path(root_dir)
        self.dump_path = self.root_path / f'checkpoint.{dump_suffix}'
        self.max_keep = max_keep
        self.epoch = 0
        self.checkpoints = deque()
        self.target_dict = target_dict

    def __repr__(self):
        repr_str = f'{type(self).__name__} ('
        repr_str += f'root={self.root_path}, max_keep={self.max_keep}, '
        repr_str += f'current_epoch={self.epoch}, checkpoints={self.checkpoints}, '
        repr_str += f'managed_keys={self.target_dict.keys()}, '
        repr_str += ')'
        return repr_str

    @classmethod
    def load_model_from_dir(cls, root_dir):
        """Create a model from previous checkpoints"""
        new_checkpoint = cls('', {}, root_dir)
        state_dict = new_checkpoint.get_checkpoint()
        model = cls._create_model_from_state_dict(state_dict)
        return model

    @classmethod
    def _create_model_from_state_dict(cls, state_dict):
        model = build_model(state_dict['hparams'])
        model.load_state_dict(state_dict['model'])
        return model

    @classmethod
    def load_model_from_checkpoint(cls, checkpoint):
        """Create a model form an exact checkpoint"""
        state_dict = torch.load(checkpoint)
        model = cls._create_model_from_state_dict(state_dict)
        return model

    def state_dict(self):
        state_dict = {k: v.state_dict() for k, v in self.target_dict.items()}
        state_dict['epoch'] = self.epoch
        state_dict['checkpoints'] = self.checkpoints
        state_dict['hparams'] = self.hparams
        return state_dict

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        hparams = state_dict['hparams']
        if self.hparams != hparams:
            logger.warning(f'The hyper-parameters of checkpoint ({self.hparams}) and file ({hparams}) differ')
        self.hparams = hparams
        try:
            self.checkpoints = state_dict['checkpoints']
        except KeyError:
            logger.warning('Please Use newer version of asr for better checkpoint')

        for name, target in self.target_dict.items():
            try:
                target.load_state_dict(state_dict[name])
            except KeyError:
                logger.warning('Name {} is not found in checkpoint file {}'.format(
                    name, self.root_path
                ))

    def init_model(self, init_path, from_kaldi=False):
        """Init the model parameters by state dict"""
        logger.info('Load pre-trained model {}'.format(init_path))
        if from_kaldi:
            state_dict = trans_kaldi.load_kaldi_model(init_path)
        else:
            state_dict = torch.load(init_path)['model']
        model = self.target_dict['model']
        model_dict = model.state_dict()
        pretrained_dict = dict()
        for name, param in state_dict.items():
            if (name in model_dict) and (model_dict[name].shape == param.shape):
                logger.debug(f'Loading parameter {name}')
                pretrained_dict[name] = param
            else:
                logger.warning(f'Unexpected keys or unmatched shape in pre-trained model: {name}')

        logger.debug(f'Parameters ({pretrained_dict.keys() - model_dict.keys()}) in init_model is not loaded')
        logger.debug(f'Parameters ({model_dict.keys() - pretrained_dict.keys()}) is not initialized')

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    def get_checkpoint(self, epoch=None):
        """Get the checkpoint state dict of given epoch"""
        checkpoint_path = self._get_checkpoint_path(epoch)
        state_dict = torch.load(checkpoint_path)
        return state_dict

    def _get_checkpoint_path(self, epoch=None):
        """Get the checkpoint file of given epoch

        It will return the last checkpoint of previous run if epoch
        is not specified

        Args:
            epoch (int, optional):

        Return:
            checkpoint_path (Path)
        """
        if epoch is None:
            checkpoint_path = self.root_path / 'checkpoint'
        else:
            checkpoint_path = self.root_path / 'checkpoint-{}'.format(epoch)
        return checkpoint_path

    def _link_final(self):
        """Link the checkpoint to the one of given epoch"""
        dest = self.checkpoints[-1]
        final_checkpoint = self._get_checkpoint_path()
        rm_file(final_checkpoint)
        final_checkpoint.symlink_to(dest.resolve())

    def save(self, epoch):
        """Save the target dict with extra epoch info"""
        self.epoch = epoch
        checkpoint_path = self._get_checkpoint_path(epoch)
        self.checkpoints.append(checkpoint_path)
        if len(self.checkpoints) > self.max_keep:
            rm_file(self.checkpoints.popleft())
        torch.save(self.state_dict(), checkpoint_path)
        self._link_final()

    def dump_exist(self):
        return Path(self.dump_path).is_file()

    def save_dump(self):
        logger.info(f'Saving dump file to {self.dump_path}')
        torch.save(self.state_dict(), self.dump_path)

    def remove_dump(self):
        if self.dump_exist():
            logger.info(f'Removing dump file {self.dump_path}')
            rm_file(self.dump_path)

    def load_epoch(self, epoch):
        state_dict = self.get_checkpoint(epoch)
        self.load_state_dict(state_dict)

    def load_best(self, load_scheduler=True):
        """Load the best checkpoint

        Args:
            load_scheduler (bool): whether to load the scheduler from
                checkpoint. It should be set to False at training, True at
                Initialization.
        """
        if self.dump_exist():
            state_dict = torch.load(self.dump_path)
            load_path = self.dump_path
        else:
            state_dict = self.get_checkpoint()
            load_path = self._get_checkpoint_path()
        epoch = state_dict['epoch']
        log_info = f'Resuming training from {load_path}, epoch {epoch}'

        if load_scheduler:
            total_step = state_dict['lr_scheduler']['_step_count']
            log_info += f', total step {total_step:,d}'
        else:
            del state_dict['lr_scheduler']

        logger.warning(log_info)
        self.load_state_dict(state_dict)