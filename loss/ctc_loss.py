import numpy as np
import editdistance as ed

import torch

from .loss import Loss, add_loss


def distance(xs, ys, xlen, ylen, blank):
    def path(x):
        prev = blank
        seq = []
        for i in x:
            if i not in [blank, prev]:
                seq.append(i)
            prev = i
        return seq
    err = 0
    for x, y, xl, yl in zip(xs, ys, xlen, ylen):
        err += ed.eval(path(x[:int(xl)]), y[:int(yl)])
    return err


@add_loss('ctc')
class CTCLoss(Loss):
    """A wrapper for WarpCTC Loss

    It hides some configurations from WarpCTCLoss.

    Args:
        output (Field): model output, with tensor of shape(B, T, D)
        data_batch (dict): the data_batch from batch loader.
                           label (Field), feat (Field)

    """

    def __init__(self, size_average=False, blank=0):
        super().__init__()
        # Lazy init
        from warpctc_pytorch import CTCLoss as WarpCTCLoss
        self.loss_kernel = WarpCTCLoss(size_average=size_average, blank=blank)

    def forward(self, output, data_batch):
        assert output.length is not None, 'Model must produce Field output with length for ctc'
        label = data_batch['label'].tensor
        label_length = data_batch['label'].length
        output_length = output.length
        output = output.tensor

        label = label.cpu().numpy()
        ctc_label = torch.tensor(
            np.hstack([label[i][:int(l)]
                       for i, l in enumerate(label_length)]),
            dtype=torch.int32
        )

        output = output.transpose(0, 1)  # WarpCTC accepts T,B,N posterior
        loss = self.loss_kernel(output, ctc_label, output_length, label_length)
        loss = loss.to(output.device)
        if torch.isinf(loss): # make loss zero when it is inf
            loss.zero_()
        loss_statistics = self._get_statistics(loss, output, label, output_length, label_length)

        return loss, loss_statistics

    def _get_statistics(self, loss, output, label, output_length, label_length):
        utterances = len(label_length)
        loss_item = loss.item()  # utterance level ctc loss * utterances
        _, prediction = torch.max(output, 2)
        prediction = prediction.cpu().numpy().T
        error_labels = distance(prediction, label, output_length, label_length, 0)
        labels = label_length.sum().item()
        correct_labels = labels - error_labels
        loss_statistics = {
            'loss': loss_item,
            'correct_labels': correct_labels,
            'total_labels': labels,
            'total_frames': sum(output_length).item(),
            'utterances': utterances,
        }

        return loss_statistics

    def log_line(self, reduced_stat):
        """Convert the reduced statistics into a log line"""
        token_accuracy = reduced_stat['correct_labels'] / reduced_stat['total_labels']
        loss_per_frame = reduced_stat['loss'] / reduced_stat['total_frames']
        return f'Token acc: {token_accuracy * 100:.2f}, Loss: {loss_per_frame:.2f}'
