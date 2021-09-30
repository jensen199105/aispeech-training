"""Utils to pad variable length inputs"""
import torch
import numpy as np

from ..field import Field


def _pad_arrs(arrs, pad_axis=0, pad_val=0):
    """Padding a list of variable length tensor into a minibatch form

    Args:
        arrs (list[numpy.ndarray]): list of arrays for padding.
        pad_axis (int, optional): the axis for padding. Defaults to 0.
        pad_val (int, optional): padding value. Defaults to 0.

    Returns:
        ret (torch.Tensor): data in the minibatch, Shape is (B, ...)
        original_length (torch.Tensor): the length of the data, shape is (B,)

    """
    assert isinstance(arrs[0], np.ndarray), 'Expects numpy.ndarray but get type {}'.format(type(arrs[0]))

    original_length = [ele.shape[pad_axis] for ele in arrs]
    max_size = max(original_length)

    ret_shape = list(arrs[0].shape)
    ret_shape[pad_axis] = max_size
    ret_shape = (len(arrs), ) + tuple(ret_shape)

    ret = np.full(shape=ret_shape, fill_value=pad_val, dtype=arrs[0].dtype)
    for i, arr in enumerate(arrs):
        if arr.shape[pad_axis] == max_size:
            ret[i] = arr
        else:
            slices = [slice(None) for _ in range(arr.ndim)]
            slices[pad_axis] = slice(0, arr.shape[pad_axis])
            slices = [slice(i, i + 1)] + slices
            ret[tuple(slices)] = arr

    ret = torch.from_numpy(ret)
    original_length = torch.tensor(original_length, dtype=torch.int32)
    return Field(ret, original_length)


def pad_multi_arrs(arrs_list, pad_val=0):
    """Pad multiple lists of arrays into list of batches and lengths

    Args:
        arrs_list (List[(feat_A, feta_B, ...)]): a list of tuple of
        multiple data fields
        pad_val (int, optional): constant padding value

    Returns:
        batch_arr_list (tuple(batch_A, batch_B, ...)): the padded batch
        for each field
        batch_arr_len_list (tuple(length_A, length_B, ...)): the length
        for each field
    """
    fields = [_pad_arrs(arrs, pad_val=pad_val) for arrs in zip(*arrs_list)]
    return fields
