import ast
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def rm_file(path):
    """Remove the path with-out raise FileNotFoundError"""
    # TODO: python3.8 supports missing_ok flag
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def reduce_number(num):
    """Reduce any python number using pytorch's distributed communication"""
    tensor = torch.cuda.FloatTensor([num]).cuda()
    dist.all_reduce(tensor)
    return tensor.item()


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')

    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def str2bool(v):
    if type(v) is bool:
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def str2dict(string):
    """Safely convert the string from cmd to python dict

    There are many methods to convert string to dict, using
    `ast.literal_eval` is prefered to `eval`.

    Ref:
    https://stackoverflow.com/questions/988228/convert-a-string-representation-of-a-dictionary-to-a-dictionary
    """
    return ast.literal_eval(string)


def pretty_time(elapsed):
    hours, left = divmod(elapsed, 3600)
    minutes, seconds = divmod(left, 60)
    return '{}:{:0>2}:{:0>2}'.format(int(hours), int(minutes), int(seconds))


def zero_pad_concat(inputs, val=0, dtype='float'):
    max_t = max(inp.shape[0] for inp in inputs)
    shape = (len(inputs), max_t) + inputs[0].shape[1:]
    input_mat = np.full(shape, val, dtype=dtype)
    for e, inp in enumerate(inputs):
        input_mat[e, :inp.shape[0]] = inp
    return input_mat


def read_log_prior(path):
    """Read the log output unit prior from disk file

    The prior is usually named as ``label.counts`` in kaldi, it should
    be **self-normalized in log-domain** .

    Args:
        path (Path): path to the piror file

    Returns:
        np.Array: log prior vector
    """

    path = Path(path)
    with path.open() as f:
        arr = f.readline().strip().split()
        arr.remove('[')
        arr.remove(']')
        log_prior = np.array(arr, dtype=np.float32)

    return log_prior
