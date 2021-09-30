"""Batch collector gathers multiple samples and put them into one batch"""
from ...utils.dynload_factory import _factory_add, _factory_build


class Collector():
    """Collect the data samples and generate a minibatch.

    Any collectors should implement the following methods to be an
    available collector.
    """

    def __init__(self):
        pass

    def add_sample(self, sample):
        """Add one sample to the collector

        Usually collector should cache the samples until they can make a batch.

        Args:
            sample (tuple(utt_id, payload)): one data record
        """
        raise NotImplementedError

    def get_batches(self):
        """Get the available batches

        Returns:
            list[:class:`~asr.data.field.Batch`]: the data batch list
        """
        raise NotImplementedError

    def get_remaining(self):
        """Get the remaining batches

        Returns:
            list[:class:`~asr.data.field.Batch`]: the data batch list
        """
        raise NotImplementedError

    def is_batch_ready(self):
        """Determine if the collector can generate batches

        Returns:
            boolean: batch ready flag
        """
        raise NotImplementedError


_collector_mapping = {}

add_collector = _factory_add(_collector_mapping, force_base_class=Collector)
build_collector = _factory_build(_collector_mapping)
