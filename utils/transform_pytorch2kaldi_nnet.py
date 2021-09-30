#!/usr/bin/env python
# encoding: utf-8
"""transform model from pytorch format to kaldi format"""
import numpy as np
import torch
from ..model import build_model


def load_pytorch_model(checkpoint):
    state = torch.load(checkpoint, map_location='cpu')
    model = state['model']
    params = state['hparams']
    np.set_printoptions(threshold=1000000000, linewidth=1000000000, suppress=True, precision=9)
    return model, params


def process_fsmn_layer(pytorch_layer):

    nproj = pytorch_layer.proj.out_features
    ninp = pytorch_layer.hidden.in_features
    nhid = pytorch_layer.hidden.out_features
    lo = pytorch_layer.fsmn.l_order
    ro = pytorch_layer.fsmn.r_order
    ls = pytorch_layer.fsmn.l_stride
    rs = pytorch_layer.fsmn.r_stride
    clip = 6 if '6' in pytorch_layer.activation.__str__() else 0
    skip = 1 if pytorch_layer.skip_connection is not None else 0
    metadata_1 = f'<Dfsmn> {nproj} {ninp} <LOrder> {lo} <ROrder> {ro}'
    metadata_2 = f'<LStride> {ls} <RStride> {rs} <HidSize> {nhid}'
    metadata_3 = f'<LearnRateCoef> 1 <BiasLearnRateCoef> 1 <MaxNorm> 0 <ClipGradient> 0'
    metadata_4 = f'<Skip> {skip} <Clip> {clip} '
    metadata = f'{metadata_1} {metadata_2} {metadata_3} {metadata_4}'
    filter_weight = pytorch_layer.fsmn.filter.detach().numpy()
    filter_weight = str(filter_weight).replace('[', '').replace(']', '')
    hidden_weight = pytorch_layer.hidden.weight.detach().numpy()
    hidden_weight = str(hidden_weight).replace('[', '').replace(']', '')
    hidden_bias = pytorch_layer.hidden.bias
    hidden_bias = hidden_bias.view(hidden_bias.shape[0], -1).detach().numpy()
    hidden_bias = str(hidden_bias.transpose()).replace('[', '').replace(']', '')
    proj_weight = pytorch_layer.proj.weight.detach().numpy()
    proj_weight = str(proj_weight).replace('[', '').replace(']', '')
    fsmn_layer = metadata + '\n [ ' + filter_weight + ' ]\n [ ' + hidden_weight + ' ] \n [ ' \
        + hidden_bias + ' ] \n [ ' + proj_weight + ' ] \n'
    return fsmn_layer


def process_affine_relu_layer(pytorch_layer, clip):

    nhid = pytorch_layer.out_features
    ninp = pytorch_layer.in_features
    metadata_1 = f'<AffineTransformRelu> {nhid} {ninp} '
    metadata_2 = f'<LearnRateCoef> 1 <BiasLearnRateCoef> 1 <MaxNorm> 0'
    metadata_3 = f'<ClipGradient> 0 <Clip> {clip} '
    metadata = f'{metadata_1}{metadata_2} {metadata_3}'
    hidden_wight = pytorch_layer.weight.detach().numpy()
    hidden_wight = str(hidden_wight).replace('[', '').replace(']', '')
    line = pytorch_layer.bias.shape[0]
    hidden_bias = pytorch_layer.bias.view(line, -1).detach().numpy().transpose()
    hidden_bias = str(hidden_bias).replace('[', '').replace(']', '')
    affine_layer = metadata + ' \n [ ' + hidden_wight + ' ] \n [ ' + hidden_bias + ' ] \n'
    return affine_layer


def process_affine_layer(pytorch_layer):

    nhid = pytorch_layer.out_features
    ninp = pytorch_layer.in_features
    metadata_1 = f'<AffineTransform> {nhid} {ninp}\n'
    metadata_2 = f'<LearnRateCoef> 1 <BiasLearnRateCoef> 1 <MaxNorm> 0 '
    metadata = f'{metadata_1}{metadata_2}'
    hidden_weight = str(pytorch_layer.weight.detach().numpy()).replace('[', '').replace(']', '')
    line = pytorch_layer.bias.shape
    hidden_bias = pytorch_layer.bias.view(line, -1).detach().numpy().transpose()
    hidden_bias = str(hidden_bias).replace('[', '').replace(']', '')
    affine_layer = metadata + '\n [ ' + hidden_weight + ' ] \n [ ' + hidden_bias + ' ] \n'
    return affine_layer


def process_cnn_layer(pytorch_layer, inp):
    """transform cnn to kaldi format

    .. warning::

        The cnn hidden weight format differs from kaldi, use with caution.
        In pytorch is arranged by (nchannel, 1, filter_0, filter_1),
        while in  kaldi arranged by (nchannel, filter_0 * filter_1).

    Args:
        nchannel(int): num of cnn channels
        filter_size(tuple): cnn kernel size
    """

    inp_0 = inp[0]
    inp_1 = inp[1]
    filter_0 = pytorch_layer[0].kernel_size[0]
    filter_1 = pytorch_layer[0].kernel_size[1]
    cin = inp_0 * inp_1
    cout0 = inp_0 - filter_0 + 1
    cout1 = inp_1 - filter_1 + 1
    nchannel = pytorch_layer[0].out_channels
    maxstride_0 = pytorch_layer[1].stride[0]
    maxstride_1 = pytorch_layer[1].stride[1]
    maxpool_0 = pytorch_layer[1].kernel_size[0]
    maxpool_1 = pytorch_layer[1].kernel_size[1]
    cout = cout0 * cout1 * nchannel
    metadata_1 = f'<Convolutional2DComponentFast> {cout} {cin}\n<LearnRateCoef> 1 <BiasLearnRateCoef> 1'
    metadata_2 = f'<FmapXLen> {inp_0} <FmapYLen> {inp_1} <FiltXLen> {filter_0} <FiltYLen> {filter_1}'
    metadata_3 = f'<FiltXStep> 1 <FiltYStep> 1 <ConnectFmap> 0 <Filters>  [ \n'
    metadata = f'{metadata_1} {metadata_2} {metadata_3}'
    filter_size = filter_0 * filter_1
    hidden_weight = pytorch_layer[0].weight.detach().numpy()
    hidden_weight = str(hidden_weight.reshape(nchannel, filter_size)).replace('[', '').replace(']', '')
    hidden_bias = pytorch_layer[0].bias.detach().numpy()
    hidden_bias = str(hidden_bias).replace('[', '').replace(']', '')
    poolin = (inp_0 - filter_0 + 1) * (inp_1 - filter_1 + 1) * nchannel
    fmapxlen = inp_0 - filter_0 + 1
    fmapylen = inp_1 - filter_1 + 1
    poolout = int((fmapxlen - maxstride_0 + 1) * (fmapylen / maxstride_1) * nchannel)
    metadata_4 = f'<MaxPooling2DComponentFast> {poolout} {poolin}\n<FmapXLen> {fmapxlen} <FmapYLen> {fmapylen}'
    metadata_5 = f'<PoolXLen> {maxpool_0} <PoolYLen> {maxpool_1} <PoolXStep> {maxstride_0} <PoolYStep> {maxstride_1}'
    metadata_6 = f'<Relu> {poolout} {poolout}\n'
    metadata_7 = f'{metadata_4} {metadata_5} {metadata_6}'
    cnn_layer = metadata + hidden_weight + ' ]\n <Bias> [ ' + hidden_bias + ' ]\n' + metadata_7
    return cnn_layer, fmapylen, nchannel


def process_lstm_layer(pytorch_layer):
    lstmpcell = pytorch_layer.lstmpcell
    projection_in = lstmpcell.projection.in_features
    projection_out = lstmpcell.projection.out_features
    lstm_in = lstmpcell.gates_layer.in_features - lstmpcell.projection.out_features
    lstm_gate = lstmpcell.gates_layer.weight.detach().numpy().T
    lstm_gate_in = lstm_gate[0: lstm_in]
    lstm_gate_proj = lstm_gate[lstm_in: lstmpcell.gates_layer.in_features]
    lstm_bias = lstmpcell.gates_layer.bias.detach().numpy()
    lstm_bias = str(lstm_bias.reshape(1, lstm_bias.shape[0])).replace('[', '').replace(']', '')
    lstm_i_peep = lstmpcell.i_peephole_weight.detach().numpy()
    lstm_i_peep = str(lstm_i_peep.reshape(1, lstm_i_peep.shape[0])).replace('[', '').replace(']', '')
    lstm_f_peep = lstmpcell.f_peephole_weight.detach().numpy()
    lstm_f_peep = str(lstm_f_peep.reshape(1, lstm_f_peep.shape[0])).replace('[', '').replace(']', '')
    lstm_o_peep = lstmpcell.o_peephole_weight.detach().numpy()
    lstm_o_peep = str(lstm_o_peep.reshape(1, lstm_o_peep.shape[0])).replace('[', '').replace(']', '')
    lstm_proj = lstmpcell.projection.weight.detach()
    lstm_proj = str(lstm_proj.numpy()).replace('[', '').replace(']', '')
    metadata_1 = f'<LearnRateCoef> 1 <BiasLearnRateCoef> 1 <MaxNorm> 0 '
    metadata_2 = f'<CellDim> {projection_in} <ClipGradient> 5  [ \n'
    metadata_3 = f'<LstmProjectedStreamsFast> {projection_out} {lstm_in}\n'
    lstm_gate_in = str(lstm_gate_in.T).replace('[', '').replace(']', '')
    lstm_gate_proj = str(lstm_gate_proj.T).replace('[', '').replace(']', '')
    lstm_layer = metadata_3 + metadata_1 + metadata_2 + lstm_gate_in + ' ]\n [ ' + lstm_gate_proj \
        + ' ]\n [ ' + lstm_bias + ' ]\n [ ' + lstm_i_peep + ' ]\n [ ' + lstm_f_peep + ' ]\n [ ' \
        + lstm_o_peep + ' ]\n [ ' + lstm_proj + ' ]\n'
    return lstm_layer


def convert_fsmn(params, model, checkpoint):
    hparams = {
        'nhid': 2048,
        'nproj': 512,
        'nvocab': 121,
        'ro': 10,
        'ls': 1,
        'rs': 1,
        'ndnn': 2
    }
    hparams.update(params)
    fsmn_nnet = ''
    checkpoint = build_model(hparams)
    checkpoint.load_state_dict(model)
    # process layer fsmn
    for fsmn in checkpoint.fsmn.fsmns:
        kaldi_fsmn = process_fsmn_layer(fsmn)
        fsmn_nnet = fsmn_nnet + kaldi_fsmn
    # process affine
    dnn_nnet = ''
    for i in range(checkpoint.dnn.__len__() // 2):
        if '6' in checkpoint.dnn[i * 2 + 1].__str__():
            clip = 6
        else:
            clip = 0
        kaldi_dnn = process_affine_relu_layer(checkpoint.dnn[i * 2], clip)
        dnn_nnet = dnn_nnet + kaldi_dnn
    # process layer out
    kaldi_out = process_affine_layer(checkpoint.output)
    softmax = f'<Softmax> {checkpoint.output.out_features} {checkpoint.output.out_features}\n'
    kaldi_nnet = '<Nnet>\n' + fsmn_nnet + dnn_nnet + kaldi_out + softmax + '</Nnet>\n'
    return kaldi_nnet


def convert_cld(params, model, checkpoint):
    hparams = {
        'nhid': 1536,
        'nvocab': 121,
        'nproj': [320, 320, 448, 448],
        'nchannel': 256,
        'lstm_inp': 320,
        'nlayer': 4,
        'lstm_out': 2048,
        'inp_size': [11, 40],
        'maxpool_size': [1, 3],
        'filter_size': [9, 8],
        'maxpool_stride': [1, 3]
    }
    hparams.update(params)
    checkpoint = build_model(hparams)
    checkpoint.load_state_dict(model)
    # process cnn layer
    kaldi_cnn, fmapyLen, nchannel = process_cnn_layer(checkpoint.cnn, hparams['inp_size'])
    # process projection layer
    # Warning : project layer weight is arranged by (proj_input.out_features, nchannel , fmapyLen),
    #            while in kaldi is arranged by (proj_input.out_features, fmapyLen , nchannel),
    #            so here we transpose nchannel and fmapyLen and then reshape the weight to 2D
    #            ( proj_input.out_features, fmapyLen * nchannel );
    with torch.no_grad():
        checkpoint.proj_input.weight.copy_(
            checkpoint.proj_input.weight.view(
                checkpoint.proj_input.out_features, nchannel, fmapyLen
            ).transpose(1, 2).contiguous().view(checkpoint.proj_input.out_features, -1)
        )
    project_layer = process_affine_layer(checkpoint.proj_input)
    # process lstm layer
    lstm_layer = ''
    for lstm in checkpoint.lstm.lstm.lstm.rnn:
        kaldi_lstm = process_lstm_layer(lstm)
        lstm_layer = lstm_layer + kaldi_lstm
    # process lstm_out layer
    kaldi_dnn = process_affine_layer(checkpoint.lstm.output_layer)
    hidden = checkpoint.lstm.output_layer.out_features
    dnn_relu = f'<Relu> {hidden} {hidden}\n'
    # preocess out layer
    kaldi_out = process_affine_layer(checkpoint.dnn[1])
    softmax = f'<Softmax> {checkpoint.dnn[1].out_features} {checkpoint.dnn[1].out_features}\n'
    kaldi_nnet = '<Nnet>\n' + kaldi_cnn + project_layer + lstm_layer + kaldi_dnn \
        + dnn_relu + kaldi_out + softmax + '</Nnet>\n'
    return kaldi_nnet


def trans_pytorch2kaldi(checkpoint):
    """transform pytorch format to kaldi format

    Args :
        checkpoint: the path of the well trained pytorch model
    """

    model, params = load_pytorch_model(checkpoint)
    model_name = params['name']
    if 'FSMN' == model_name:
        print('The model is fsmn')
        return convert_fsmn(params, model, checkpoint)
    elif 'CLD' == model_name:
        print('The model is cld')
        return convert_cld(params, model, checkpoint)
    else:
        raise ValueError(f'Model type unsupported')
