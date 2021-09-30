#!/usr/bin/env python
# encoding: utf-8
"""transform model from kaldi format into pytorch format"""
import re
import numpy as np
import torch


def _parse_line(line):
    """Get params from lines"""

    if line == '<Nnet>' or line == '</Nnet>':
        return 'nnet', None

    # parameters
    tmp = re.search(r'^[\[|\s]*([0-9\-.e]+ )+[\s|\]]*', line)
    if tmp:
        line = line.split()
        if line[0] == '[':
            line = line[1:]
        if line[-1] == ']':
            line = line[:-1]
        return 'params', [float(x) for x in line]

    # Maxpooling config
    tmp = re.search(
        r'<FmapXLen> (\d+) <FmapYLen> (\d+) <PoolXLen> (\d+) <PoolYLen> (\d+) <PoolXStep> (\d+) <PoolYStep> (\d+) .*',
        line)
    if tmp:
        input_size = (int(tmp.group(1)), int(tmp.group(2)))
        pool_size = (int(tmp.group(3)), int(tmp.group(4)))
        stride_size = (int(tmp.group(5)), int(tmp.group(6)))
        maxpool_out_h = (input_size[0] - pool_size[0]) / stride_size[0] + 1
        maxpool_out_w = (input_size[1] - pool_size[1]) / stride_size[1] + 1
        return 'maxpool_config', maxpool_out_h * maxpool_out_w

    # Dfsmn, AffineTransformRelu, AffineTransform, Softmax
    tmp = re.search(r'<(\S*)> (\d+) (\d+)', line)
    if tmp:
        name = tmp.group(1)
        name = name.lower()[:min(len(name), 20)]
        return name, [int(tmp.group(2)), int(tmp.group(3))]

    # Cnn config filter_size stride
    tmp = re.search(
        r'<FmapXLen> (\d+) <FmapYLen> (\d+) <FiltXLen> (\d+) <FiltYLen> (\d+) <FiltXStep> (\d+) <FiltYStep> (\d+) .*',
        line)
    if tmp:
        input_size = (int(tmp.group(1)), int(tmp.group(2)))
        filter_size = (int(tmp.group(3)), int(tmp.group(4)))
        stride_size = (int(tmp.group(5)), int(tmp.group(6)))
        cnn_out_h = (input_size[0] - filter_size[0]) / stride_size[0] + 1
        cnn_out_w = (input_size[1] - filter_size[1]) / stride_size[1] + 1
        return 'cnn_config', [filter_size, int(cnn_out_h * cnn_out_w)]

    # Dfsmn config, filter_size, hidden_size
    tmp = re.search(r'<LOrder> (\d+) <ROrder> (\d+) .* <HidSize> (\d+) .*', line)
    if tmp:
        filter_size = int(tmp.group(1)) + 1 + int(tmp.group(2))
        return 'dfsmn_config', [filter_size, int(tmp.group(3))]

    # Lstm config celldim
    tmp = re.search(r'.* <CellDim> (\d+) .*', line)
    if tmp:
        return 'lstm_config', [int(tmp.group(1))]
    if line == '[':
        return 'params_begin', None

    raise Exception(f'Not implemented line pattern {line}')


def _process_cell_param(load_file, celldim, outdim):

    weight_list = []
    for _ in range(celldim):
        state, info = _parse_line(load_file.readline().strip())
        if state != 'params' or len(info) != outdim:
            raise Exception(f'Format incorrect, expected {outdim} but got {len(info)}')
        else:
            weight_list.append(np.array(info))

    return np.stack(weight_list, 0)


def _process_cnn(file_handle, cnn_odim):
    """cnn layer"""

    load_lines = file_handle

    line = load_lines.readline().strip()
    state, info = _parse_line(line)
    if state == 'cnn_config':
        filter_size = info[0]
        cnn_out_size = info[1]

    channels = int(cnn_odim / cnn_out_size)

    cnn_weight = _process_cell_param(load_lines, channels, filter_size[0] * filter_size[1])

    line = load_lines.readline().strip()
    if line.split()[0] == "<Bias>":
        line = " ".join(line.split()[1:])
    else:
        raise Exception(f"Format incorrect, expected cnn_config with <FmapXLen>, but got line {line}")

    _, info = _parse_line(line)
    cnn_bias = np.array(info)

    param_dict = {
        'cnn_weight': cnn_weight,
        'cnn_bias': cnn_bias,
    }

    return param_dict, filter_size, channels


def _process_maxpool(file_handle):

    load_lines = file_handle

    line = load_lines.readline().strip()
    state, info = _parse_line(line)
    if state == 'maxpool_config':
        out_dim = info
    else:
        raise Exception(f"Format incorrect, expected maxpool_config with <FmapXLen>, but got line {line}")

    return out_dim


def _process_dfsmn(file_handle, idim, odim):
    """fsmn layer"""

    load_lines = file_handle

    line = load_lines.readline().strip()
    state, info = _parse_line(line)
    if state == 'dfsmn_config':
        filter_size = info[0]
        hidden_size = info[1]
    else:
        raise Exception(f"Format incorrect, expected dfsmn_config with <LOrder>, but got line {line}")

    filter_mat = _process_cell_param(load_lines, filter_size, odim)

    load_lines.readline()  # read the line '['

    x_hidden = _process_cell_param(load_lines, hidden_size, idim)

    # bias
    state, info = _parse_line(load_lines.readline().strip())
    bias = np.array(info)

    load_lines.readline()  # read the line '['

    # hidden_p
    hidden_p = _process_cell_param(load_lines, odim, hidden_size)

    param_dict = {
        'filter_mat': filter_mat,
        'x_hidden': x_hidden,
        'bias': bias,
        'hidden_p': hidden_p,
    }

    return param_dict


def _process_lstms(file_handle, lstm_idim, lstm_odim):
    """lstm layer"""

    load_lines = file_handle

    state, info = _parse_line(load_lines.readline().strip())
    if state == 'lstm_config':
        cell_dim = info[0]
    else:
        raise Exception(f'Format incorrect, expected lstm_config with <CellDim>, but got line {info}')

    weight_gate_input = _process_cell_param(load_lines, cell_dim * 4, lstm_idim)

    load_lines.readline()  # read the line '['

    weight_gate_hidden = _process_cell_param(load_lines, cell_dim * 4, lstm_odim)

    _, info = _parse_line(load_lines.readline().strip())
    bias_gate = np.array(info)

    # peephole weight
    peephole_weights = []
    for _ in range(3):
        _, info = _parse_line(load_lines.readline().strip())
        peephole_weights.append(np.array(info))

    param_dict = {
        'weight_gate_input': weight_gate_input,
        'weight_gate_hidden': weight_gate_hidden,
        'bias_gate': bias_gate,
        'peephole_i_c': peephole_weights[0],
        'peephole_f_c': peephole_weights[1],
        'peephole_o_c': peephole_weights[2],
    }

    return cell_dim, param_dict


def _process_projection(file_handle, idim, odim):
    """lstm projection layer"""

    load_lines = file_handle

    load_lines.readline().strip()  # read the line '['

    proj_weights = _process_cell_param(load_lines, odim, idim)

    return proj_weights


def _process_affine(file_handler, idim, odim):
    """affinetransform layer"""

    load_lines = file_handler
    load_lines.readline().strip()  # read the head line
    weights = _process_cell_param(load_lines, odim, idim)

    state, info = _parse_line(load_lines.readline().strip())
    if state != 'params' or len(info) != odim:
        raise Exception(f"Format incorrect, expected {odim} but got {len(info)}")
    bias = np.array(info)

    param_dict = {
        'weights': weights,
        'bias': bias,
    }
    return param_dict


def read_kaldi_parameters(kaldi_model):
    """read kaldi format parameters from each layey

    Args:
        kaldi_model: opened kaldi nnet file

    Return:
        kaldi model params which saved as dict
    """

    nnet_file = kaldi_model
    parameters = []

    line = nnet_file.readline().strip()
    while line != '':
        line_state, info = _parse_line(line)

        if line_state == 'dfsmn':
            param_dict = _process_dfsmn(nnet_file, info[1], info[0])
            parameters.append(
                dict(type='dfsmn', idim=info[1], odim=info[0],
                     filter_mat=param_dict['filter_mat'],
                     x_hidden=param_dict['x_hidden'],
                     bias=param_dict['bias'],
                     hidden_p=param_dict['hidden_p'])
            )

        elif line_state == 'lstmprojectedstreams':
            cell_dim, param_dict = _process_lstms(nnet_file, info[1], info[0])
            proj_weights = _process_projection(nnet_file, cell_dim, info[0])
            parameters.append(
                dict(type='lstm', idim=info[1], cell_dim=cell_dim, odim=info[0],
                     weight_gate_input=param_dict['weight_gate_input'],
                     weight_gate_hidden=param_dict['weight_gate_hidden'],
                     bias_gate=param_dict['bias_gate'],
                     peephole_i_c=param_dict['peephole_i_c'],
                     peephole_f_c=param_dict['peephole_f_c'],
                     peephole_o_c=param_dict['peephole_o_c'],
                     proj_weights=proj_weights)
            )
        elif line_state == 'convolutional2dcompo':
            param_dict, filter_size, nchannel = _process_cnn(nnet_file, info[0])
            parameters.append(
                dict(type='cnn', nchannel=nchannel, filter_size=filter_size,
                     cnn_weight=param_dict['cnn_weight'],
                     cnn_bias=param_dict['cnn_bias'])
            )
        elif line_state == 'maxpooling2dcomponen':
            odim = _process_maxpool(nnet_file)
            parameters.append(
                dict(type='maxpool', odim=odim)
            )
        elif line_state == 'affinetransformrelu':
            param_dict = _process_affine(nnet_file, info[1], info[0])
            parameters.append(
                dict(type='dnnrelu', idim=info[1], odim=info[0],
                     weights=param_dict['weights'],
                     bias=param_dict['bias'])
            )
        elif line_state == 'affinetransform':
            param_dict = _process_affine(nnet_file, info[1], info[0])
            parameters.append(
                dict(type='dnn', idim=info[1], odim=info[0],
                     weights=param_dict['weights'],
                     bias=param_dict['bias'])
            )

        line = nnet_file.readline().strip()

    if line_state != 'nnet':
        raise Exception("The file is ended incomplete.")

    return parameters


def _load_lstm_model(parameters):
    """transform lstm model"""
    state_dict = {}
    lstm = -1
    for param in parameters:
        if param['type'] == 'lstm':
            lstm += 1
            torch_model = f'lstm.lstm.rnn.{lstm}.lstmpcell'
            state_dict[f'{torch_model}.gates_layer.weight'] = torch.cat(
                [torch.from_numpy(param['weight_gate_input']),
                 torch.from_numpy(param['weight_gate_hidden'])], dim=1)
            state_dict[f'{torch_model}.gates_layer.bias'] = torch.from_numpy(param['bias_gate'])
            state_dict[f'{torch_model}.i_peephole_weight'] = torch.from_numpy(param['peephole_i_c'])
            state_dict[f'{torch_model}.f_peephole_weight'] = torch.from_numpy(param['peephole_f_c'])
            state_dict[f'{torch_model}.o_peephole_weight'] = torch.from_numpy(param['peephole_o_c'])
            state_dict[f'{torch_model}.projection.weight'] = torch.from_numpy(param['proj_weights'])
        elif param['type'] == 'dnn':
            state_dict[f'output_layer.weight'] = torch.from_numpy(param['weights'])
            state_dict[f'output_layer.bias'] = torch.from_numpy(param['bias'])

    return state_dict


def _load_cld_model(parameters):
    """transform cld model"""

    state_dict = {}
    lstm = 0
    aff = 0
    nchannel = 0
    filter_size = ()
    maxpool_odim = 0
    for param in parameters:
        if param['type'] == 'lstm':
            torch_model = f'lstm.lstm.lstm.rnn'
            state_dict[f'{torch_model}.{lstm}.lstmpcell.gates_layer.weight'] = torch.cat([
                torch.from_numpy(param['weight_gate_input']), torch.from_numpy(param['weight_gate_hidden'])
            ], dim=1)
            state_dict[f'{torch_model}.{lstm}.lstmpcell.gates_layer.bias'] = torch.from_numpy(param['bias_gate'])
            state_dict[f'{torch_model}.{lstm}.lstmpcell.i_peephole_weight'] = torch.from_numpy(param['peephole_i_c'])
            state_dict[f'{torch_model}.{lstm}.lstmpcell.f_peephole_weight'] = torch.from_numpy(param['peephole_f_c'])
            state_dict[f'{torch_model}.{lstm}.lstmpcell.o_peephole_weight'] = torch.from_numpy(param['peephole_o_c'])
            state_dict[f'{torch_model}.{lstm}.lstmpcell.projection.weight'] = torch.from_numpy(param['proj_weights'])
            lstm += 1
        elif param['type'] == 'cnn':
            nchannel = int(param['nchannel'])
            filter_size = param['filter_size']
            state_dict[f'cnn.0.weight'] = torch.from_numpy(param['cnn_weight']).view(
                nchannel, 1, filter_size[0], filter_size[1])
            state_dict[f'cnn.0.bias'] = torch.from_numpy(param['cnn_bias'])
        elif param['type'] == 'maxpool':
            maxpool_odim = int(param['odim'])
        elif param['type'] == 'dnn' and aff == 0:
            odim = int(param['odim'])
            state_dict[f'proj_input.weight'] = torch.from_numpy(
                param['weights']
            ).view(odim, maxpool_odim, nchannel).transpose(2, 1).reshape(odim, -1)
            state_dict[f'proj_input.bias'] = torch.from_numpy(param['bias'])
            aff += 1
        elif param['type'] == 'dnn' and aff == 1:
            state_dict[f'lstm.output_layer.weight'] = torch.from_numpy(param['weights'])
            state_dict[f'lstm.output_layer.bias'] = torch.from_numpy(param['bias'])
            aff += 1
        elif param['type'] == 'dnn' and aff == 2:
            state_dict[f'dnn.1.weight'] = torch.from_numpy(param['weights'])
            state_dict[f'dnn.1.bias'] = torch.from_numpy(param['bias'])
            aff += 1

    return state_dict


def _load_fsmn_model(parameters):
    """Transform fsmn model"""
    state_dict = {}
    dfsmn = -1
    dnn = -2
    for param in parameters:
        if param['type'] == 'dfsmn':
            dfsmn += 1
            state_dict[f'fsmn.fsmns.{dfsmn}.hidden.weight'] = torch.from_numpy(param['x_hidden'])
            state_dict[f'fsmn.fsmns.{dfsmn}.hidden.bias'] = torch.from_numpy(param['bias'])
            state_dict[f'fsmn.fsmns.{dfsmn}.proj.weight'] = torch.from_numpy(param['hidden_p'])
            state_dict[f'fsmn.fsmns.{dfsmn}.fsmn.filter'] = torch.from_numpy(param['filter_mat'])
        elif param['type'] == 'dnnrelu':
            dnn += 2
            state_dict[f'dnn.{dnn}.weight'] = torch.from_numpy(param['weights'])
            state_dict[f'dnn.{dnn}.bias'] = torch.from_numpy(param['bias'])
        elif param['type'] == 'dnn':
            state_dict[f'output.weight'] = torch.from_numpy(param['weights'])
            state_dict[f'output.bias'] = torch.from_numpy(param['bias'])

    return state_dict


def reload_torch_model(model_type, parameters):
    """transform different models

    Args:
        model_type(str): specify model type fsmn/cld/lstm
        parameters(dict): kaldi model parameters read from each layer

    Return:
        pytorch format model state_dict which params replaced by kaldi model params
    """

    if model_type == 'fsmn':
        return _load_fsmn_model(parameters)
    elif model_type == 'lstm':
        return _load_lstm_model(parameters)
    elif model_type == 'cld':
        return _load_cld_model(parameters)

    raise ValueError(f'Model type {model_type} unsupported')


def load_kaldi_model(kaldi_nnet):
    """Transform model from kaldi txt format into pytorch format

    Args:
        kaldi_nnet(str): kaldi txt format nnet file path with model type like model_type://nnet_path
                         eg: fsmn:///mnt/lustre/aifs/fgfs/users/kaldi_nnet.txt

    Return:
        pytorch format model state_dict which params replaced by kaldi model params
    """

    model_type, nnet_path = kaldi_nnet.split("://")

    with open(nnet_path, 'r') as f:
        params = read_kaldi_parameters(f)

    return reload_torch_model(model_type, params)
