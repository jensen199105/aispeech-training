import torch
import numpy as np

from .collector import Collector, add_collector
from ..field import Batch, Field

# FIXME: it may exceed the frame limit
# FIXME: only supports single-view single-task currently
# FIXME: only supports feat and ali
@add_collector('concat')
class ConcatCollector(Collector):
    """Concatenate multiple utterances into one big utterance

    Args:
        frame_limit (int): maximum number of frames in a batch
    """

    def __init__(self, frame_limit):
        super().__init__()
        self.frame_limit = frame_limit
        self.streams = []
        self.batch_frames = 0

    def __repr__(self):
        return f'{self.__class__.__name__} (frame_limit={self.frame_limit})'

    def add_sample(self, sample):
        _, payload = sample
        feat_list, ali_list, _, _ = payload
        self.streams.append((feat_list[0], ali_list[0]))
        self.batch_frames += feat_list[0].shape[0]

    def get_batches(self):
        feat, ali = zip(*self.streams)
        feat = Field(
            torch.from_numpy(np.concatenate(feat, 0)).unsqueeze(0),
            torch.LongTensor([self.batch_frames]).unsqueeze(0),
        )
        label = Field(
            torch.from_numpy(np.concatenate(ali, 0)).unsqueeze(0),
            torch.LongTensor([self.batch_frames]).unsqueeze(0),
        )
        batch = Batch({
            'feat': feat,
            'label': label,
        })
        self.streams = []
        self.batch_frames = 0
        return [batch]

    def get_remaining(self):
        return self.get_batches()

    def is_batch_ready(self):
        return self.batch_frames >= self.frame_limit