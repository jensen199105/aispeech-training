import math
import logging

from .lr_scheduler import LRScheduler, decision, add_scheduler

logger = logging.getLogger(__name__)


@add_scheduler('kaldi')
class KaldiLRScheduler(LRScheduler):
    """Kaldi style learning rate scheduler

    It decays the weight once relative improvement is less than 0.5%, and keeps decaying
    in the following epochs. The training is terminated if the learning rate is decaying
    and relative improvement becomes less than 0.1%.

    The learning rate can be gradually warmed up linearly by specify non-zero ``warmup_round``
    and ``warmup_batches_per_round``.

    Parameters:
        - warmup_round (int): #rounds the scheduler spends to increase the learning rate
        - warmup_batches_per_round (int): #batches a round contains
        - rel_stop (float): relative improvement threshold for stopping the training
        - rel_decay (float): relative improvement threshold for starting to decay lr
    """

    def __init__(self, optimizer, warmup_round=0, warmup_batches_per_round=0, rel_stop=0.001, rel_decay=0.005):
        self._last_metric = math.inf
        self.decay_factor = 1
        self.warmup_round = warmup_round
        self.warmup_batches_per_round = warmup_batches_per_round
        self.rel_stop = rel_stop
        self.rel_decay = rel_decay
        super().__init__(optimizer)

    def _step(self):
        """Warming-up learning rate at the beginning of training.

        It supports learning rate warming up if both ``warmup_round`` and
        ``warmup_batches_per_round`` and specified.
        """
        # Make sure the learning rate is correct even when resuming from a checkpoint
        if self._step_count >= self.warmup_batches_per_round * self.warmup_round:
            self._apply_decay()
        elif self.warmup_round > 0 and self._step_count % self.warmup_batches_per_round == 0:
            warmup_rd = self._step_count // self.warmup_batches_per_round
            for param_group in self.optimizer.param_groups:
                warmup_lr = param_group['initial_lr'] / self.warmup_round
                next_lr = warmup_rd * warmup_lr
                param_group['lr'] = next_lr
            logger.debug(
                f'Warmup No.{warmup_rd} : lr={self.get_lr_str()}'
            )

        self._step_count += 1

    def _step_epoch(self, metric):
        """Kaldi style lr update policy"""
        metric = float(metric)
        if math.isnan(metric):
            metric = math.inf
        rel_improve = (self._last_metric - metric) / self._last_metric
        self.last_epoch += 1

        decisions = []

        # stop
        if self.is_decaying() and rel_improve < self.rel_stop:
            self.lr_decay(factor=0)
            decisions.append(decision.STOP)
            if rel_improve <= 0:
                decisions.append(decision.REJECT)
            else:
                decisions.append(decision.ACCEPT)
            return decisions, f'Finished, too small rel improvement ({rel_improve * 100:.2f}% < {self.rel_stop * 100}%).'

        # reject
        if rel_improve <= 0:
            self.lr_decay()
            decisions.append(decision.REJECT)
            return decisions, 'No improvement, lr decay'

        decisions.append(decision.ACCEPT)

        self._last_metric = metric
        if rel_improve < self.rel_decay:
            self.lr_decay()
            return decisions, f'Low improvement ({rel_improve * 100:.2f}% < {self.rel_decay * 100}%), lr decay'

        # halving and accept
        if self.is_decaying():
            self.lr_decay()
            return decisions, 'Continue decaying'

        return decisions, ''

    def _apply_decay(self):
        """Apply the decay factor onto optimizer

        This method can be called at any time because we maintain full states
        to make learning rate decay robust (hopefully)
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.decay_factor * param_group['initial_lr']

    def is_decaying(self):
        """If decay_factor is not 1, the lr is decaying"""
        return self.decay_factor != 1

    def lr_decay(self, factor=0.5):
        """Increase the decay factor to do lr decay"""
        self.decay_factor *= factor
        self._apply_decay()

    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]
