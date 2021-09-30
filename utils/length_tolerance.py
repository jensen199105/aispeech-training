import logging

import torch

from ..data.field import Field

logger = logging.getLogger(__name__)


def fix_mismatch(label_field, output_field, uids=None, length_tolerance=0):
    """Make label's length same as output's length, based on the latter's length
    Args:
        label_field (:class:`~asr.data.field.Field`)
        output_field (:class:`~asr.data.field.Field`)

    Returns:
        fixed_label (:class:`~asr.data.field.Field`)
        fixed_output (:class:`~asr.data.field.Field`)

    """

    label, label_lens = label_field.tensor, label_field.length
    output, output_lens = output_field.tensor, output_field.length
    if output.size(0) != label.size(0):
        raise RuntimeError(f'The model output size ({output.size(0)}) must '
                           f'match label size ({label.size(0)}) for batch dimension. '
                           'Please check if you transposed the output correctly.')
    batch_filter = [True ] * len(output_lens)
    T_label, T_output = label.size(1), output.size(1)
    if T_label > T_output:
        label = label[:, :T_output]
        label_lens[label_lens >T_output] = T_output
    elif T_label < T_output:
        label = torch.nn.functional.pad(label, [0, T_output-T_label], 'constant', 0)

    for b, (label_len, expect_len) in enumerate(zip(label_lens, output_lens)):
        diff_len = expect_len - label_len
        if 0 <= torch.abs(diff_len) <= length_tolerance:
            if diff_len >= 0:
                label[b, label_len:expect_len] = label[b, label_len -1] # edge padding
            else:
                label[b, label_len:expect_len] = -1 # do not calculate cross-entropy
        else:
            batch_filter[b] = False
            uid = uids[b] if uids is not None else None
            logger.warning(f'Utterance {uid}, label & feature length mismatch,'
                           f'{label_len} & {expect_len}')

    fixed_label = Field(label[batch_filter], output_lens[batch_filter])
    fixed_output = Field(output[batch_filter], output_lens[batch_filter])
    return fixed_label, fixed_output