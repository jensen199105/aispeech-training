import torch.nn as nn

from asr.data.field import Field

from .lstm import LSTM
from .model import Model, add_model


@add_model('CLD')
class CLD(Model):
    """Convolution + LSTM + DNN

    Args:
        inp_size(tuple): input feature dims
        nchannel(int): num of cnn channels
        filter_size(tuple): cnn kernel size
        maxpool_size(tuple): maxpooling kernel size
        maxpool_stride(tuple): pooling stride
        lstm_inp(int): input dim for lstm layer
        nhid(int): num of hidden units for lstm
        nproj(tuple): lstm projection for each layer
        subsample(tuple): num of frame inner-skip in each lstm layers, 1 means no skip
        nlayer(int): num of lstm layers
        lstm_out(int): output dim after lstm
        nvocab(int): final output dim
    """

    def __init__(self, inp_size=(11, 40), nchannel=256, filter_size=(9, 8),
                 maxpool_size=(1, 3), maxpool_stride=(1, 3),
                 lstm_inp=320, nhid=1536, nproj=(320, 320, 448, 448), subsample=(1, 1, 1, 1), nlayer=4, lstm_out=2048,
                 nvocab=121):
        super().__init__()

        if (inp_size[0] - filter_size[0] + 1 - maxpool_size[0]) % maxpool_stride[0] != 0 or \
           (inp_size[1] - filter_size[1] + 1 - maxpool_size[1]) % maxpool_stride[1] != 0:
            maxpooling_input = (inp_size[0] - filter_size[0] + 1, inp_size[1] - filter_size[1] + 1)
            raise ValueError(f'maxpooling_input {maxpooling_input} minus maxpool_size {maxpool_size} ' \
                             f'must be divisible by pool_stride {maxpool_stride}')

        self.inp_size = inp_size
        self.cnn = nn.Sequential(
            nn.Conv2d(1, nchannel, filter_size),
            nn.MaxPool2d(kernel_size=maxpool_size, stride=maxpool_stride),
            nn.ReLU6(),
        )

        self.maxpool_h = ((inp_size[0] - filter_size[0] + 1 - maxpool_size[0]) // maxpool_stride[0]) + 1
        self.maxpool_w = ((inp_size[1] - filter_size[1] + 1 - maxpool_size[1]) // maxpool_stride[1]) + 1
        self.maxpool_out = int(self.maxpool_h * self.maxpool_w * nchannel)

        self.proj_input = nn.Linear(self.maxpool_out, lstm_inp)
        self.lstm = LSTM(lstm_inp, nhid, list(nproj),
                         nlayer=nlayer, nvocab=lstm_out, subsample=subsample)
        self.dnn = nn.Sequential(
            nn.ReLU6(),
            nn.Linear(lstm_out, nvocab),
        )

    def init_parameters(self):
        self.lstm.init_parameters()

    def grad_post_processing(self):
        self.lstm.grad_post_processing()

    def forward(self, batch):
        out = batch['feat'].tensor.cuda()
        B, T, _ = out.size()
        out = out.view(B * T, 1, self.inp_size[0], self.inp_size[1])
        out = self.cnn(out).view(B, T, self.maxpool_out)
        out = self.proj_input(out)
        batch['feat'].tensor = out
        out = self.lstm(batch)
        out_length = out.length
        out = self.dnn(out.tensor)
        return Field(out, out_length)
