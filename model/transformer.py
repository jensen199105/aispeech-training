"""Attention is all you need!!"""
import math

import torch
import torch.nn as nn

from .model import Model, add_model
from ..data.field import Field


class PositionalEncoding(nn.Module):
    r"""Positional embedding copied from pytorch/examples

    Inject some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.

    .. math::
        PosEncoder(pos, 2i) = sin(\frac{pos}{10000^{2i/d_{model}}}) \\
        PosEncoder(pos, 2i+1) = cos(\frac{pos}{10000^{2i/d_{model}}}) \\
        \text{where } pos \text{ is the word position and }i \text{ is the embed idx}

    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).

    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function

        Args:
            x: the sequence fed to the positional encoder model (required).

        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]

        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def generate_local_attention_mask(left_context=0, right_context=0, max_len=5000):
    """Generate the local attention mask for a given window size

    Sometimes we want to limit the attention window because in some tasks like asr
    , the output is highly correlated to local context rather than global information

    Args:
        - left_context (int): the length of left context
        - right_context (int): the length of right context
        - max_len (int): the maximum possible length of one sequence

    Returns:
        mask (:class:`~torch.FloatTensor`): the generated mask matrix of shape
            :math:`(S, S)` where S is max_len. Mask matrix is `(Target, Source)`.
            See :class`~torch.nn.MultiHeadAttention` and
            https://pytorch.org/docs/stable/nn.html#torch.nn.Transformer.forward
    """
    mask = torch.zeros(max_len, max_len).fill_(0)
    # Fill
    for diag_offset in range(-left_context, right_context + 1):
        diag_size = max_len - abs(diag_offset)
        if diag_size > 0:
            mask += torch.ones(diag_size).diagflat(diag_offset)
        else:
            raise ValueError(f'Context too long (-{left_context}, {right_context}) '
                             f'exceeding maximum sequence length ({max_len})')
    # pytorch takes -inf as False mask and 0 as True mask
    return mask.log()


@add_model('Transformer')
class Transformer(Model):
    """Transformer encoder serving as sequential model

    This module borrrows some ideas from speech-transformer

    Ref: https://github.com/kaituoxu/Speech-Transformer
    """

    def __init__(self, ninp, nhid, nlayer, nvocab,
                 nhead=16, context=None, max_len=2048):
        super().__init__()
        self.input_proj = nn.Linear(ninp, nhid)
        self.layer_norm_in = nn.LayerNorm(nhid)
        self.positional_embedding = PositionalEncoding(nhid, max_len=max_len)
        transformer_layer = nn.TransformerEncoderLayer(d_model=nhid, nhead=nhead)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=nlayer)
        self.out_proj = nn.Linear(nhid, nvocab)

        # Local attention mask
        self.context = context
        if context is not None:
            attention_mask = generate_local_attention_mask(context[0], context[1], max_len)
            self.register_buffer('attention_mask', attention_mask)

    def forward(self, batch):
        # Transformer accepts T,B,D input
        tensor_in = batch['feat'].tensor.transpose(0, 1).cuda()
        max_seq_len = tensor_in.size(0)
        out = self.input_proj(tensor_in)
        out = self.positional_embedding(out)
        out = self.layer_norm_in(out)
        if self.context is not None:
            mask = self.attention_mask[:max_seq_len, :max_seq_len]
        else:
            mask = None
        out = self.transformer(out, mask=mask)
        out = self.out_proj(out).transpose(0, 1)
        return Field(out, batch['feat'].length)
