# pylint: disable=E1130
import math
import logging

import torch
import torch.nn as nn

from ..data.field import Field

from .model import Model, add_model
from .module import build_activation


__all__ = ['FSMN']


logger = logging.getLogger(__name__)


@add_model('FSMN')
class FSMN(Model):
    """Feedforward Sequential Memory Network(FSMN)

    DFSMN acoustic model, followed by arbitrary number of DNN layer.

    Parameters:
        ninp (int): input dimension size
        nhid (int): FSMN hidden size
        nproj (int): FSMN projection size
        nvocab (int): output dimension for softmax
        skip (str, optional): skip connection between two DFSMN layers,
                              choices are {'res', 'highway', None}
        nlayer (int): number of DFSMN layers
        ndnn (int): number of the dnn layers following dfsmn (including
                    output layer)
        lo (int): left order
        ro (int): right order
        ls (int): left stride
        rs (int): right stride
        max_norm (int): maximum gradient norm
        activation (str): ``relu`` ``relu2`` or ``relu6``. The default activation
                          in FSMN and DNN, empirically ``relu6`` trains more stable
                          than ``relu``, especially in small dataset.
                          ``relu2`` is recommended for local model.
        clip_weight (float): clip weight during training, :math:`|weight|<=clip_weight`
    """
    def __init__(self, ninp, nhid, nproj, nvocab,
                 skip='res', nlayer=8, ndnn=2, lo=20, ro=0, ls=1,
                 rs=1, max_norm=5000, activation='relu6', clip_weight=math.inf, kernel_res=True):
        super().__init__()
        self.max_norm = max_norm
        self.clip_weight = clip_weight
        if self.clip_weight <= 0:
             raise ValueError(f'clip_weight must greater than 0, not {self.clip_weight}.')

        self.activation = build_activation(activation, inplace=True)

        # NOTE: it's a private package, lazy_init
        from fsmn import DFSMN
        self.fsmn = DFSMN(ninp, nhid, nproj, lo, ro, ls, rs, nlayer, 0, skip,
                          batch_first=False, activation=self.activation, kernel_res=kernel_res)

        if ndnn > 0:
            dnns = [nn.Linear(nproj, nhid), self.activation]
            for _ in range(ndnn - 1):
                dnns.extend([nn.Linear(nhid, nhid), self.activation])
            output_in = nhid
        elif ndnn == 0:
            dnns = []
            output_in = nproj
        else:
            raise ValueError('This implement needs ndnn >= 0 followed by FSMN layers')

        self.dnn = nn.Sequential(*dnns)
        self.output = nn.Linear(output_in, nvocab)
        # DFSMN and all other modules will init the parameters themselves

    def _clip_weight(self):
        for param in self.parameters():
            torch.clamp_(param.data, min=-self.clip_weight, max=self.clip_weight)

    def grad_post_processing(self):
        """Clip the accumulated norm of all gradients to max_norm"""
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)
        if norm >= self.max_norm:
            logger.debug(f'Norm overflow: {norm}')

    def forward(self, batch):
        # (B, T, D)
        self._clip_weight()
        xs = batch['feat'].tensor.transpose(0, 1)
        length = batch['feat'].length

        xs = xs.cuda()
        cuda_length = length.cuda()

        xs = self.fsmn(xs, cuda_length)
        xs = self.dnn(xs)
        xs = self.output(xs)
        # NOTE: The transpose MUST be placed after the DNN layer
        # remains to be solved
        xs = xs.transpose(0, 1).contiguous()

        return Field(xs, length)
