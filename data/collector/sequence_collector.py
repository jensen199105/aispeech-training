import logging
from collections.abc import Iterable

from ..field import Batch
from .collector import Collector, add_collector
from .padding import pad_multi_arrs

logger = logging.getLogger(__name__)


@add_collector('sequence')
class SequenceCollector(Collector):
    """Generic variable length sequence collector.

    It pads a batch of sequences with variable length to the maximum length.

    Parameters:
        minibatch_size (int): the size of a minibatch, max number of utterances in one batch
        frame_limit (int, optional): the limit of frames in a minibatch, including padding.
    """

    def __init__(self, minibatch_size, frame_limit=25000):
        super(SequenceCollector, self).__init__()
        self.minibatch_size = minibatch_size
        self.frame_limit = frame_limit
        self.streams = []
        self.stream_max_length = 0
        self.batch = None

    def __repr__(self):
        return f'{self.__class__.__name__} (minibatch_size={self.minibatch_size}, frame_limit={self.frame_limit})'

    def _generate_minibatch(self):

        assert self.streams, 'Error: call generate minibatch when streams is empty'

        uid_list, payload_list = zip(*self.streams)
        batch_feat_list, batch_ali_list, batch_ivec_list, batch_lat_list = zip(*payload_list)

        batch_feats = pad_multi_arrs(batch_feat_list, pad_val=0)
        batch_labels = pad_multi_arrs(batch_ali_list, pad_val=-1)
        # ivecs could possibily be empty list
        batch_ivecs = pad_multi_arrs(batch_ivec_list, pad_val=0)
        batch_lats = [list(lat) for lat in zip(*batch_lat_list)]

        # extra namespace is used for multi-task / multi-view, and extra features such as
        # lattices and ivectors
        # TODO: delete the keys with empty values
        extra = {
            'feat': batch_feats[1:],
            'label': batch_labels[1:],
            'ivec': batch_ivecs,
            'lattices': batch_lats,
        }

        label = batch_labels[0] if batch_labels else None

        self.batch = Batch({
            'uid': uid_list,
            'feat': batch_feats[0],
            'label': label,
            'extra': extra,
        })
        self.streams = []

    def add_sample(self, sample):
        uid, payload = sample
        feat_list, _, _, _ = payload
        for item in payload:
            assert isinstance(item, Iterable), 'Data sample expects tuple of Iterable, but get {}'.format(type(item))

        sample_length = feat_list[0].shape[0]
        if sample_length > self.frame_limit:
            logger.warning(f"Utterance {uid} length over frame limit ({sample_length}, {self.frame_limit})")
            return
        # The max length if current stream is appended to stream buffer
        maybe_stream_max_length = max(sample_length, self.stream_max_length)
        num_stream = len(self.streams)
        if num_stream == self.minibatch_size or maybe_stream_max_length * (1 + num_stream) > self.frame_limit:
            self._generate_minibatch()
            self.stream_max_length = 0

        self.streams.append(sample)
        self.stream_max_length = max(sample_length, self.stream_max_length)

    def is_batch_ready(self):
        return self.batch is not None

    def get_batches(self):
        assert self.batch is not None, 'Minibatch not ready'
        batch = self.batch
        self.batch = None
        return [batch]

    def get_remaining(self):
        if self.streams:
            self._generate_minibatch()
            return self.get_batches()
        else:
            return None
