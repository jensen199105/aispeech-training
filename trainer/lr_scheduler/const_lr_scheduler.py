import logging

from .lr_scheduler import LRScheduler, decision, add_scheduler

logger = logging.getLogger(__name__)


@add_scheduler('const')
class ConstLRScheduler(LRScheduler):
    """Implement constant learning rate scheduler.

    It will use the constant learning rate to complete the training process.

    We may use this scheduler in MMI training.
    """

    def __init__(self, optimizer):
        super().__init__(optimizer)

    def _step_epoch(self, _):
        self.last_epoch += 1
        return [decision.ACCEPT], 'Always accept'

    def _step(self):
        self._step_count += 1

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]
