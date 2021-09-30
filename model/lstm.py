import torch.nn as nn

from asr.data.field import Field

from .model import Model, add_model
from .module.kaldi_lstmp import KaldiLSTMP, StreamLSTM


@add_model('LSTM')
class LSTM(Model):
    def __init__(self, ninp, nhid, nproj, nlayer, nvocab, subsample=None, dropout=0, step_size=None):
        super(LSTM, self).__init__()

        lstmp = KaldiLSTMP(ninp, nhid, nproj, nlayer, subsample, dropout, step_size)
        self.lstm = StreamLSTM(lstmp)
        self.output_layer = nn.Linear(nproj[-1], nvocab)

        self.total_sub = 1
        for sub in lstmp.subsample:
            self.total_sub *= sub

        self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            # init weight
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.uniform_(param.data)
            # init bias
            if 'bias' in name:
                param.data.fill_(0.)
            # init lstm forget gate bias
            if 'lstm' in name and 'bias' in name:
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.)

    def grad_post_processing(self):
        for name, param in self.named_parameters():
            if 'lstm' in name and param.grad is not None:
                param.grad.data.clamp_(min=-5, max=5)

    def forward(self, batch):
        xs = batch['feat'].tensor
        length = batch['feat'].length
        output_len = (length + self.total_sub - 1) // self.total_sub
        label = batch['label']
        if label is not None:
            skipped_label = label.tensor[:, ::self.total_sub]
            skipped_label_len = (label.length + self.total_sub - 1) // self.total_sub
            batch['skipped_label'] = Field(skipped_label, skipped_label_len)

        xs = xs.transpose(0, 1).cuda()
        ys = self.lstm(xs, batch.get('new_stream', None))

        output = self.output_layer(ys).transpose(0, 1).contiguous()
        return Field(output, output_len)
