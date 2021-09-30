import torch

from .lr_scheduler import LRScheduler, decision, add_scheduler


@add_scheduler('noam')
class NoamLRScheduler(LRScheduler):
    """Implement Noam learning rate scheduler

    This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.

    This scheduler is used by espnet's optimizer. The source code is copied from allennlp (Apache 2.0)

    Parameters:
        - model_size (int): The hidden size parameter which dominates the number of parameters in your model.
        - warmup_steps (int): The number of steps to linearly increase the learning rate.
        - factor (float, optional): The overall scale factor for the learning rate decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_size: int,
        warmup_steps: int,
        factor: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.model_size = model_size
        super().__init__(optimizer, last_epoch=last_epoch)

    def _step(self) -> None:
        self._step_count += 1
        for param_group, learning_rate in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = learning_rate

    def _step_epoch(self, _):
        self.last_epoch += 1
        return [decision.ACCEPT], 'Always accept'

    def get_lr(self):
        scale = self.factor * (
            self.model_size ** (-0.5) * min(self._step_count ** (-0.5), self._step_count * self.warmup_steps ** (-1.5))
        )
        return [scale for _ in self.base_lrs]
