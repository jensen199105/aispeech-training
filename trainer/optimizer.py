import logging
from enum import Enum

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required  # pylint: disable=no-name-in-module

logger = logging.getLogger(__name__)


class BMethod(Enum):
    MA = 'MA'
    BMUF_CBM = 'BMUF-NBM'
    BMUF_NBM = 'BMUF-CBM'


class BMOptimizer(Optimizer):
    """A block momentum optimizer to perform model average

    Block momentum gathers the gradients from multiple workers and updates in a momentum-SGD
    like fashion.

    Ref:
        https://docs.microsoft.com/en-us/cognitive-toolkit/multiple-gpus-and-machines#6-block-momentum-sgd

    .. warning::
        the global learning rate and momentum is different from local ones.

    Parameters:
        - params (Iterable of torch.nn.Parameters): the parameters to do block
            momentum update
        - method (str): block momentum algorithms, choices are {'MA', 'BMUF-CBM', 'BMUF-NBM'}.
        - momentum (float): the global momentum.
        - lr (float): the global learning rate.
    """

    def __init__(self, params, method=required, momentum=None, lr=None):
        if lr is not None and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum is not None and momentum < 0.0:
            logger.critical(f'Invalid global moementum specified: {momentum}, fall back to default value')
            momentum = None

        method = BMethod(method)
        self.world_size = dist.get_world_size()
        miracle_lr = 1
        miracle_momentum = 1 - 1 / self.world_size
        if lr is None:
            lr = miracle_lr
        if momentum is None:
            momentum = miracle_momentum

        defaults = dict(lr=lr, momentum=momentum, method=method)
        super().__init__(params, defaults)

    def load_state_dict(self, state_dict: dict) -> None:
        for group in state_dict['param_groups']:
            group['method'] = BMethod(group['method'])
        super().load_state_dict(state_dict)

    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        for group in state_dict['param_groups']:
            group['method'] = group['method'].value
        return state_dict

    def step(self):
        """Perform a single optimization step for distributed training

        There's no closure argument because the loss is evaluated in
        single node optimizer

        Returns:
            valid (boolean): if the updated parameters are all valid,
                i.e. no inf or nan
        """
        if self.world_size == 1:
            return True

        for group in self.param_groups:
            method = group['method']
            if method == BMethod.MA:
                self._avg_step(group)
            else:
                self._bm_step(group)
            return self._check_valid()

    def _check_valid(self):
        for group in self.param_groups:
            for p in group['params']:
                if not torch.isfinite(p).all():
                    return False
        return True

    def _bm_step(self, group):
        grad_scale = group['lr']
        momentum = group['momentum']
        bm_method = group['method']
        for p in group['params']:

            # Init optimizer state
            param_state = self.state[p]
            if 'prev_weight' not in param_state:
                param_state['prev_weight'] = p.data.clone()
            if 'momentum_buffer' not in param_state:
                param_state['momentum_buffer'] = torch.zeros_like(p.data)
            prev_p = param_state['prev_weight']
            v_p = param_state['momentum_buffer']

            # average all the models
            dist.all_reduce(p)

            # block gradient computing
            block_grad = p.data.div_(self.world_size).sub_(prev_p).neg_()
            # v_t
            v_p.mul_(momentum).add_(grad_scale, block_grad)

            if bm_method == BMethod.BMUF_CBM:
                p.data = prev_p - v_p
            elif bm_method == BMethod.BMUF_NBM:
                p.data = prev_p - (1 + momentum) * v_p
            # update cache_p
            prev_p.copy_(p.data)

    def _avg_step(self, group):
        for p in group['params']:
            dist.all_reduce(p)
            p.data.div_(self.world_size)


class EarlyStop(Exception):
    """Current worker needs to stop before batch consumed"""


class DistributedOptimizer:
    """An abstraction of synchronized distributed training"""
    def __init__(self, local_optimizer, global_optimizer, sync_interval=1):
        self.local_optimizer = local_optimizer
        self.global_optimizer = global_optimizer
        self.sync_interval = sync_interval
        self.accumulated_interval = 0
        self.sync_count = 0

    def __repr__(self):
        return f'{self.__class__.__name__}(interval={self.sync_interval})'

    def clear(self):
        """Clear the optimizer state before every epoch starts"""
        self.local_optimizer.zero_grad()
        # Reset the momentum buffer and previous parameter cache to zero
        self.local_optimizer.state.clear()
        # self.global_optimizer.state.clear()

    def step(self, interval=1):
        """Step the optimizer"""
        self.local_optimizer.step()
        self.local_optimizer.zero_grad()
        self.accumulated_interval += interval
        if self.accumulated_interval > self.sync_interval:
            self.accumulated_interval = 0
            from ..utils.common import reduce_number
            early_stop = reduce_number(0)
            if early_stop > 0:
                logger.info('Other rank suggests to end the epoch!')
                raise EarlyStop()

            self.global_optimizer.step()
            # NOTE: this should only be applied to SGD with momentum
            self.local_optimizer.state.clear()  # Reset the momentum of SGD
            logger.debug('Merge NO.{}'.format(self.sync_count))
            self.sync_count += 1

    def force_step(self):
        """Force to run the distributed optimizer

        Usually it's for synchronizing the remaining batches which does not
        meet a sync interval, at the end of one epoch
        """
        self.global_optimizer.step()
        self.local_optimizer.state.clear()
