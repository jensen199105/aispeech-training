# pylint: disable=E1130
import math
import logging

import torch
import torch.nn as nn

from asr.data.field import Field

from .model import Model

__all__ = ['GRUMODEL']


logger = logging.getLogger(__name__)


class GRUMODEL(Model):
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
        activation (str): ``relu`` ``relu2`` or ``relu6``. Use ``relu6`` instead of
                          ``relu``, empirically ``relu6`` trains more stable than
                          ``relu``, especially in small dataset.
                          ``relu2`` is recommended when you train a local task.
        clip_weight (float): clip weight during training, |weight|<=clip_weight
    """
    def __init__(self, ninp, nhid, nproj, nvocab,
                 skip='res', nlayer=1, ndnn=1, lo=0, ro=0, ls=1,
                 rs=1, max_norm=5000, activation='relu6', clip_weight=math.inf):
        super(GRUMODEL, self).__init__()
        self.max_norm = max_norm
        self.clip_weight = clip_weight
        self.nlayer = nlayer
        self.nhid = nhid
        if self.clip_weight <= 0:
            raise ValueError(f'clip_weight must greater than 0, not {self.clip_weight}.') 
        if activation == 'relu6':
            self.activation = nn.ReLU6(True)
        elif activation == 'relu2':
            self.activation = nn.Hardtanh(min_val=0, max_val=2, inplace=True)
        else:
            self.activation = nn.ReLU(True)

        # NOTE: it's a private package, lazy_init
        
        if ndnn > 0:
            dnns = [nn.Linear(ninp, nhid), self.activation]
            for _ in range(ndnn - 1):
                dnns.extend([nn.Linear(nhid, nhid), self.activation])
            output_in = nhid
        elif ndnn == 0:
            dnns = []
            output_in = nhid
        else:
            raise ValueError('This implement needs ndnn >= 0 followed by FSMN layers')
        
        #self.gru1 = nn.GRU(ninp,nhid, nlayer,batch_first=False)
	#self.gru2 = nn.GRU(10, 20, 2)
	#self.gru3 = nn.GRU(10, output_in, 2)
        self.dnn = nn.Sequential(*dnns)
        #self.dnn = nn.Linear(ninp,nhid,batch_first=False)
        self.tanh= nn.Tanh()
        self.gru1 = nn.GRU(nhid, nhid, nlayer,self.activation)
        self.output = nn.Linear(output_in, nvocab)
        self.sigmoid= nn.Sigmoid()
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
	#h0 = Variable(torch.zeros(self.nlayer, xs.size(0), self.nhid).cuda())

        #xs = self.dnn(xs, cuda_length)
        xs = self.dnn(xs)
        xs = self.tanh(xs)
        xs, hn = self.gru1(xs,hn)
	# xs = self.gru2(xs)
	#xs = self.gru3(xs)
        xs = self.output(xs)
        xs = self.sigmoid(xs)
        if torch.isnan(xs).sum() > 0:
            print(batch['feat'].tensor)
            print(xs)
            for param in self.parameters():
                print(param)
            asdsadsa
        # NOTE: The transpose MUST be placed after the DNN layer
        # remains to be solved
        xs = xs.transpose(0, 1).contiguous()

        return Field(xs, length)
