import math
import torch

from .collector import Collector, add_collector
from ..field import Batch
from .padding import pad_multi_arrs


@add_collector('stream')
class StreamCollector(Collector):
    """Utterance stream collector for LSTM-like model

    It feeds utterances into multiple streams and maintains a ``new_stream``
    indicator for each uttterance.

    .. warning:: ONLY use for LSTM-like model in CE training

    Parameters:
        stream_size (int): number of streams in one batch
        step_size (int): utterance length in one batch
    """

    def __init__(self, stream_size, step_size):
        super(StreamCollector, self).__init__()
        self.stream_size = stream_size
        self.step_size = step_size
        self.streams = [None] * self.stream_size
        self.streams_len = [0] * self.stream_size
        self.new_stream = [0] * self.stream_size
        self.batches = []

    def __repr__(self):
        return f'{self.__class__.__name__} (stream_size={self.stream_size}, step_size={self.step_size})'

    def _generate_minibatch(self):

        min_length = min(self.streams_len)
        num_batch = math.ceil(min_length / self.step_size)

        for i in range(num_batch):
            batch_feat_list = []
            batch_ali_list = []
            for feat_list, ali_list in self.streams:
                feats = [feat[:self.step_size] for feat in feat_list]
                alis = [ali[:self.step_size] for ali in ali_list]
                for k in range(len(feat_list)):
                    feat_list[k] = feat_list[k][self.step_size:]
                for k in range(len(ali_list)):
                    ali_list[k] = ali_list[k][self.step_size:]
                batch_feat_list.append(feats)
                batch_ali_list.append(alis)

            batch_feats = pad_multi_arrs(batch_feat_list, pad_val=0)
            batch_labels = pad_multi_arrs(batch_ali_list, pad_val=-1)
            # extra namespace is used for multi-task / multi-view
            # TODO: delete the keys with empty values
            extra = {
                'feat': batch_feats[1:],
                'label': batch_labels[1:],
            }

            if i == 0:
                new_stream = self.new_stream
            else:
                new_stream = [0] * self.stream_size
            new_stream = torch.tensor(new_stream)

            label = batch_labels[0] if batch_labels else None

            batch = Batch({
                'feat': batch_feats[0],
                'label': label,
                'new_stream': new_stream,
                'extra': extra,
            })

            self.batches.append(batch)

        self.new_stream = [0] * self.stream_size
        for s in range(self.stream_size):
            self.streams_len[s] = len(self.streams[s][0][0])
            if self.streams_len[s] == 0:
                self.streams[s] = None

    def add_sample(self, sample):
        _, payload = sample
        feat_list, ali_list, ivec_list, lat_list = payload
        assert len(ivec_list) == 0, 'Stream dataloader does net support ivector yet.'
        assert len(lat_list) == 0, 'Stream dataloader does net support lattice yet.'
        feat_len = set()
        ali_len = set()
        for feat in feat_list:
            feat_len.add(len(feat))
        for ali in ali_list:
            ali_len.add(len(ali))
        assert len(feat_len) == 1, 'Multi feature length mismatch!'
        assert len(ali_len) == 1, 'Multi label length mismatch!'

        # CE loss has fixed length mismatch
        # assert feat_len.pop() == ali_len.pop(), 'Feature and Label length mismatch!'

        for s in range(self.stream_size):
            if self.streams[s] is None:
                self.streams[s] = (feat_list, ali_list)
                self.streams_len[s] = len(feat_list[0])
                self.new_stream[s] = 1
                break
        generate_batch = True
        for s in range(self.stream_size):
            if self.streams[s] is None:
                generate_batch = False
                break
        if generate_batch:
            self._generate_minibatch()

    def is_batch_ready(self):
        return len(self.batches) > 0

    def get_batches(self):
        assert self.batches, 'Minibatch not ready'
        batches = self.batches
        self.batches = []
        return batches

    def get_remaining(self):
        if self.batches:
            return self.batches
        else:
            return None

