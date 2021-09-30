import torch
import torch.nn as nn

from ..utils.length_tolerance import fix_mismatch
from .loss import Loss, add_loss


@add_loss('ce')
class CrossEntropyLoss(Loss):
    r"""A warpper of original pytorch CELoss

    It simply unpacks the output and batch into specific format, and computes
    some useful metrics for logging purposes.

    Parameters:
        length_tolerance (int): Maximum tolerance of length difference of alignment
                                and feature. If length-diff within tolerance,
                                :math:`|L_{ali}-L_{feat}|<=\text{length_tolerance}`, alignment will do
                                edge padding; otherwise you will get warning and that feature & label
                                will be dropped.
    Args:
        output (tensor, B,T,C)
        data_batch (dict)
    """

    def __init__(self, length_tolerance=0, ignore_index=-1, reduction='sum'):
        super().__init__()
        self.length_tolerance = length_tolerance
        self.loss_kernel = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def extra_repr(self):
        return f'(length_tolerance): {self.length_tolerance}'

    def forward(self, output, data_batch):
        if 'skipped_label' in data_batch:
            label_field = data_batch['skipped_label']
        else:
            label_field = data_batch['label']

        fixed_label, fixed_output = fix_mismatch(label_field, output, data_batch.get('uid'), self.length_tolerance)

        label, _ = fixed_label.tensor, fixed_label.length
        output = fixed_output.tensor
        output = output.reshape(-1, output.shape[-1]).float()
        label = label.cuda().view(-1)

        loss = self.loss_kernel(output, label)
        loss_statistics = self._get_statistics(loss, output, label)

        return loss, loss_statistics

    def _get_statistics(self, loss, output, label):
        loss_item = loss.item()  # utterance level ce loss * utterances
        _, prediction = torch.max(output.data, 1)
        correct_frames = torch.sum(torch.eq(label, prediction.view(-1))).item()
        frames = torch.sum(label != -1).item()
        loss_statistics = {
            'loss': loss_item,
            'correct_frames': correct_frames,
            'total_frames': frames,
        }

        return loss_statistics

    def log_line(self, reduced_stat):
        """Convert the reduced statistics into a log line"""
        frame_accuracy = reduced_stat['correct_frames'] / reduced_stat['total_frames']
        loss_per_frame = reduced_stat['loss'] / reduced_stat['total_frames']
        return f'Frame acc: {frame_accuracy * 100:.2f}, Loss: {loss_per_frame:.2f}'
