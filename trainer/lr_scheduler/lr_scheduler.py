import logging

from torch.optim.lr_scheduler import _LRScheduler as _LRSchedulerBase

from ...utils.dynload_factory import _factory_add, _factory_build

logger = logging.getLogger(__name__)


class decision:
    """A enum class for all the decisions from scheduler"""
    ACCEPT = 'accept'
    REJECT = 'reject'
    STOP = 'stop'


class LRScheduler(_LRSchedulerBase):
    """An abstract learning rate scheduler

    There are two *stepping* methods namely `step` and `step_epoch`. the `step` is for #batch
    related scheduling algorithm, while `step_epoch` is for #epoch and cv loss related scheduling.

    See the doc from pytorch for more concise descriptions.

    All learning rate schedulers should inherit this class and implement the following methods:

    Abstract methods:
        - _step(self):
        - _step_epoch(self, loss):
        - get_lr(self):

    Attributes:
        - _step_count
        - last_epoch
    """

    def __repr__(self):
        head = f'{self.__class__.__name__}('
        params = ', '.join([f'{key}={value}' for key, value in self.state_dict().items()])
        return head + params + ')'

    def get_lr_str(self):
        return ';'.join([str(lr) for lr in self.get_lr()])

    def step(self, epoch=None):
        """Step for each data batch

        This method will be called at the __init__ of _LRSchedulerBase, which means the _step_count
        will starts from 1.
        """
        return self._step()

    def _step(self):
        raise NotImplementedError

    def step_epoch(self, metric):
        """Compute and apply new learning rate given the CV loss or nothing

        This method will be called only after finishing an epoch

        Args:
            metric (number): the metric value, lower is better

        Return:
            decision (list[str]): the decision( :class:`~asr.trainer.lr_scheduler.decision` ) made by the scheduler.
                                  ACCEPT means the model of current epoch is accepted and will be saved to dist,
                                  REJECT means the model of current epoch is rejected and the model from previous
                                  epoch should be loaded, STOP means the training process should be stopped.
        """
        decisions, aux_info = self._step_epoch(metric)
        # Get (possibly) multiple learning rates from parameter groups of optimizer
        assert isinstance(decisions, list), f'The decisions is expected to be list (got {type(decisions)})'
        lr_str = self.get_lr_str()
        logger.warning(f'{decisions} ({aux_info}), lr={lr_str}')
        return decisions

    def _step_epoch(self, metric):
        '''Compute and apply new learning rate given the CV loss or nothing

        This method should be re-implemented by the child scheduler.

        Return:
            decision : See :meth:`~asr.trainer.lr_scheduler._LRScheduler.step_epoch`
            info (str): Auxiliary information
        '''
        raise NotImplementedError


_scheduler_mapping = {}

add_scheduler = _factory_add(_scheduler_mapping, force_base_class=LRScheduler)
build_scheduler = _factory_build(_scheduler_mapping)
