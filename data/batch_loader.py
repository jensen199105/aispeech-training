import logging
from functools import partial

from torch.multiprocessing import Process, Manager


logger = logging.getLogger(__name__)


class BatchLoader:
    """ Data batch loader

    Args:
        utterance_reader (:class:`asr.data.UtteranceReader`):
        collector (:class:`asr.data.collector`): data collector and batch generator
        indicate_end (bool, optional): Put a `None` into queue when data is run
                                       out. Defaults to True.
        drop_last (bool, optional): Drop the remaining data when data is running
                                    out. Defaults to True.
    """

    def __init__(self, utterance_reader, collector,
                 drop_last=False, indicate_end=True,
                 buffer_size=50):
        self.utt_reader = utterance_reader
        self.collector = collector
        self.drop_last = drop_last
        self.indicate_end = indicate_end
        self.queue = Manager().Queue(buffer_size)

    def __repr__(self):
        repr_str = f'{self.__class__.__name__} (\n'
        repr_str += f'  utterance reader: {self.utt_reader},\n'
        repr_str += f'  collector: {self.collector}\n'
        repr_str += ')'
        return repr_str

    def _run_worker(self):
        """Worker to load data in another process

        Load the data_rspecs into memory and batching them into mini-batches

        Args:
        """
        for sample in self.utt_reader:
            self.collector.add_sample(sample)

            if self.collector.is_batch_ready():
                batches = self.collector.get_batches()
                for batch in batches:
                    assert isinstance(batch, dict)
                    self.queue.put(batch)

        if not self.drop_last:
            remain_batches = self.collector.get_remaining()
            if remain_batches is not None:
                for batch in remain_batches:
                    self.queue.put(batch)

        if self.indicate_end:
            self.queue.put(None)

    def __iter__(self):
        """Batch Loader thread running function

        Return:
            batch_iterator (Iterable[Batch]): An iterable of data batches shared
                among multiple processes
        """

        worker = Process(target=self._run_worker)
        worker.start()
        # Timeout the queue iter in case loader process exits by accident
        get_batch = partial(self.queue.get, timeout=300)
        return iter(get_batch, None)

    def clear(self):
        """Clear the rest batches in the current epoch"""
        while True:
            batch = self.queue.get(timeout=300)
            if batch is None:
                break
