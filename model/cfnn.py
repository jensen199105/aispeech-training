import torch.nn as nn
import torch
import math
import logging
from asr.data.field import Field

from .lstm import LSTM
from .model import Model, add_model
from .module import build_activation

__all__ = ['CFNN']

logger = logging.getLogger(__name__)

@add_model('CFNN')

class CFNN(Model):
    """Convolution + FSMN + D_Convolution

    Args:
        inp_size(tuple): input feature dims
        nchannel(int): num of cnn channels
    """

    def __init__(self, ninp, nhid, nproj, nvocab,skip='res', nlayer=8, ndnn=2, lo=20, ro=0, ls=1,rs=1, max_norm=5000, activation='relu6', clip_weight=math.inf,inp_size=(1, 64)):
        super(CFNN,self).__init__()
        self.max_norm = max_norm
        self.clip_weight = clip_weight
        self.inp_size = inp_size
        self.activation = nn.ReLU6(True)
        self.conv1=nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(1,3),stride=(1,2),padding=(0,0),dilation=(1,1),bias=True) 
        self.conv2=nn.Conv2d(in_channels=8,out_channels=8,kernel_size=(1,3),stride=(1,2),padding=(0,0),dilation=(1,1),bias=True)
        self.conv3=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(1,3),stride=(1,2),padding=(0,0),dilation=(1,1),bias=True)
        self.conv4=nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(1,3),stride=(1,2),padding=(0,0),dilation=(1,1),bias=True)
        from fsmn import DFSMN
        self.fsmn = DFSMN(ninp, nhid, nproj, lo, ro, ls, rs, nlayer, 0, skip,
                          batch_first=False, activation=self.activation)       

        self.conv4_t = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 3), stride=(1, 2),padding=(0, 0), bias=True, dilation=(1, 1))
        self.conv3_t = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=(1, 3), stride=(1, 2),padding=(0, 0), bias=True, dilation=(1, 1))
        self.conv2_t = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(1, 3), stride=(1, 2),padding=(0, 0), bias=True, dilation=(1, 1))
        self.conv1_t = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(1, 3), stride=(1, 2),padding=(0, 0), bias=True, dilation=(1, 1),output_padding=(0, 1))
        self.relu =nn.ReLU6()
        self.sigmoid= nn.Sigmoid()

    def _clip_weight(self):
        for param in self.parameters():
            torch.clamp_(param.data, min=-self.clip_weight, max=self.clip_weight)

    def grad_post_processing(self):
        """Clip the accumulated norm of all gradients to max_norm"""
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)
        if norm >= self.max_norm:
            logger.debug(f'Norm overflow: {norm}')     
    def forward(self, batch):
        out = batch['feat'].tensor.cuda()
        length = batch['feat'].length
        cuda_length = length.cuda()
        B, T, _ = out.size()
        out = out.view(B , self.inp_size[0], T, self.inp_size[1])
        e1 =  self.relu(self.conv1(out))
        e2 =  self.relu(self.conv2(e1))
        e3 =  self.relu(self.conv3(e2))
        e4 =  self.relu(self.conv4(e3))
        out = e4.contiguous().transpose(1, 2)
        out = out.contiguous().view(out.size(0), out.size(1), -1)
        out = self.fsmn(out, cuda_length)

        out = out.contiguous().view(out.size(0), out.size(1), -1, 3)
        out = out.contiguous().transpose(1, 2)
        out = torch.cat((out, e4), dim=1)
        d4 = self.relu(torch.cat((self.conv4_t(out), e3), dim=1))
        d3 = self.relu(torch.cat((self.conv3_t(d4), e2), dim=1))
        d2 = self.relu(torch.cat((self.conv2_t(d3), e1), dim=1))
        d1 = self.relu(self.conv1_t(d2))
        output = torch.squeeze(d1, dim=1)
        if torch.isnan(output).sum() > 0:
            print(batch['feat'].tensor)
            print(output)
            for param in self.parameters():
                print(param)
            asdsadsa
        return Field(output, length)
