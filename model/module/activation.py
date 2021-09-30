import math
from functools import partial
import re

import torch.nn as nn
import torch.nn.functional as F


def build_activation(activation_spec, inplace=False):
    r"""Dynamically build activation given an activation_spec
    
    Args:
        activation_spec (str): string representation of activation function, such as ``relu``, ``mish`` etc.
        inplace: can optionally do the operation in-place. Default: ``False``.
    """

    activation_mapping = {
        'mish': Mish,
        'relu': partial(ReLUN, inplace=inplace)
    }

    upper_bound = math.inf
    # FIXME: For now assume the activation type name does not contain any number.
    upper_bound_match = re.search(r'[0-9]', activation_spec)

    if upper_bound_match:
        upper_bound_idx = upper_bound_match.span()[0]
        activation_type, upper_bound_spec = activation_spec[:upper_bound_idx], activation_spec[upper_bound_idx:]
        upper_bound = float(upper_bound_spec)
    else:
        activation_type = activation_spec

    if activation_type not in activation_mapping:
        raise ValueError(f'activation "{activation_spec}" is not supported now.')

    return activation_mapping[activation_type](upper_bound=upper_bound)


class Mish(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Mish}(x) = x * \tanh(\log(1 + \exp(x)))

        \text{Mish}(x, n) = \text{min}(\text{Mish}(x), n)

    Args:
        upper_bound: maximum value of the output's range.
    """

    def __init__(self, upper_bound=math.inf):
        super().__init__()
        self.upper_bound = upper_bound

    def extra_repr(self):
        return f'upper_bound={self.upper_bound}'

    def forward(self, x):
        x = x * (F.softplus(x).tanh())
        return F.hardtanh(x, max_val=self.upper_bound)


class ReLUN(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{ReLUN}(x, n) = \min(\max(0,x), n)

    Args:
        upper_bound: maximum value of the output's range.
        inplace: can optionally do the operation in-place. Default: ``False``
    """

    def __init__(self, upper_bound=math.inf, inplace=False):
        super().__init__()
        self.upper_bound = upper_bound
        self.inplace = inplace

    def extra_repr(self):
        return f'upper_bound={self.upper_bound}, inplace={self.inplace}'

    def forward(self, x):
        x = F.hardtanh(x, min_val=0, max_val=self.upper_bound, inplace=self.inplace)
        return x
