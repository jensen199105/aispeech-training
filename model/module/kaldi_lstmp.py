"""pytorch's Projected LSTM module"""
# pylint: disable=redefined-builtin
import torch
from torch import nn

__all__ = ['KaldiLSTMPCell', 'KaldiLSTMPLayer', 'KaldiLSTMP']


class KaldiLSTMPCell(nn.Module):
    def __init__(self, ninp, nhid, nproj):
        super(KaldiLSTMPCell, self).__init__()

        self.gates_layer = nn.Linear(ninp + nproj, nhid * 4)
        self.projection = nn.Linear(nhid, nproj, bias=False)

        # For peephole parameters
        self.i_peephole_weight = nn.Parameter(torch.Tensor(nhid))
        self.register_parameter('i_peephole_weight', self.i_peephole_weight)
        self.f_peephole_weight = nn.Parameter(torch.Tensor(nhid))
        self.register_parameter('f_peephole_weight', self.f_peephole_weight)
        self.o_peephole_weight = nn.Parameter(torch.Tensor(nhid))
        self.register_parameter('o_peephole_weight', self.o_peephole_weight)

        self.nproj = nproj
        self.nhid = nhid

    def forward(self, input, hidden=None):
        """
        :param torch.tensor xs: input (Bs, ninp)
        :param tuple of torch.tensor hidden: hx (Bs, nproj) and cx (Bs, nhid)
        """
        if hidden is None:
            hx = torch.zeros(input.size(0), self.nproj, dtype=input.dtype, device=input.device)
            cx = torch.zeros(input.size(0), self.nhid, dtype=input.dtype, device=input.device)
        else:
            hx, cx = hidden

        gates = self.gates_layer(torch.cat((input, hx), dim=1))
        # gates = self.ih_layer(input) + self.hh_layer(hx)

        # i_gate, f_gate, o_gate, cell_state = gates.chunk(4, dim=1)
        cell_state, i_gate, f_gate, o_gate = gates.chunk(4, dim=1)

        i_gate += self.i_peephole_weight.unsqueeze(0).expand_as(cx) * cx
        f_gate += self.f_peephole_weight.unsqueeze(0).expand_as(cx) * cx
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)

        cell_state = torch.tanh(cell_state)
        cy = (f_gate * cx + i_gate * cell_state).clamp_(-50, 50)

        o_gate = torch.sigmoid(o_gate + self.o_peephole_weight.unsqueeze(0).expand_as(cy) * cy)

        hy = self.projection(o_gate * torch.tanh(cy))

        return hy.clone(), (hy, cy)


class KaldiLSTMPLayer(nn.Module):
    """One layer KaldiLSTMP

    Args:
        ninp (int): size of input
        nhid (int): size of hidden
        nproj (int): size of output
    """

    def __init__(self, ninp, nhid, nproj, detach_step=None):
        super(KaldiLSTMPLayer, self).__init__()
        self.lstmpcell = KaldiLSTMPCell(ninp, nhid, nproj)
        self.detach_step = detach_step

    def forward(self, input, hidden=None):
        """Forward

        Args:
            input (torch.tensor [T, B, C]): a sequence input, time-first
            hidden (tuple(h, c)), optional): hidden. Defaults to None.
        """
        ys = []
        for t in range(input.size(0)):  # Seq-len
            out, hidden = self.lstmpcell(input[t], hidden)
            ys.append(out)
            if self.detach_step and (t + 1) % self.detach_step == 0:
                hidden[0].detach_()
                hidden[1].detach_()
        output = torch.stack(ys, dim=0)

        return output, hidden


class KaldiLSTMP(nn.Module):
    """Kaldi style LSTM with projection

    Parameters:
        - ninp (int): input dimensions
        - nhid (int): hidden dimensions
        - nproj (int): projection dimension
        - nlayer (int): number of layers
        - subsample (list[int]): frame subsampling at each layer.
            The length of subsample MUST equal to *nlayer*
        - dropout (float): dropout rate between LSTM layer, currently
            not working
        - step_size (int): BPTT length, *None* for infinite length
    """
    # FIXME: dropout is not working now
    # pylint: disable=unused-argument
    def __init__(self, ninp, nhid, nproj, nlayer,
                 subsample=None, dropout=0, step_size=None):
        super(KaldiLSTMP, self).__init__()

        nhid = nhid if isinstance(nhid, list) else [nhid]
        nproj = nproj if isinstance(nproj, list) else [nproj]
        if subsample is None:
            self.subsample = [1] * nlayer
        else:
            self.subsample = [int(x) for x in subsample]

        self.rnn = nn.ModuleList()
        for i in range(nlayer):
            input_size = ninp if i == 0 else nproj[min(i - 1, len(nproj) - 1)]
            hid_size = nhid[min(i, len(nhid) - 1)]
            output_size = nproj[min(i, len(nproj) - 1)]

            assert self.subsample[i] > 0, 'Inner skip should be greater than 0'
            if step_size is None:
                layer_step = None
            else:
                layer_step = step_size
                step_size = (step_size + self.subsample[i] - 1) // self.subsample[i]

            self.rnn.append(
                KaldiLSTMPLayer(input_size, hid_size, output_size, layer_step)
            )

    def forward(self, input, hiddens=None):
        nlayer = len(self.rnn)
        out_hiddens = [None] * nlayer
        if hiddens is None:
            hiddens = [None] * nlayer

        for i, lstmp_layer in enumerate(self.rnn):
            input, hidden = lstmp_layer(input, hiddens[i])
            sub = self.subsample[i]
            if sub > 1:
                input = input[::sub]

            out_hiddens[i] = hidden

        return input, out_hiddens


class StreamLSTM(nn.Module):
    """A hidden container to implement stream-like LSTM"""
    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm
        self.hiddens = None

    def repackage(self, hiddens, keep_state):
        if hiddens is None:
            return hiddens

        if isinstance(hiddens, list):
            mask = keep_state.view(-1, 1)
            hiddens = [(mask * h[0].detach(), mask * h[1].detach()) for h in hiddens]
        else:
            mask = keep_state.view(1, -1, 1)
            hiddens = tuple(mask * h.detach() for h in hiddens)
        return hiddens

    def forward(self, input, new_stream):
        if new_stream is not None:
            keep_state = 1 - new_stream.to(input)
            hiddens = self.repackage(self.hiddens, keep_state)
            output, self.hiddens = self.lstm(input, hiddens)
        else:
            output, _ = self.lstm(input, None)
        return output
