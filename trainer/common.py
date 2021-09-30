import time
import logging
from collections import defaultdict

import torch

from ..utils.common import reduce_number

logger = logging.getLogger(__name__)


class Metric(defaultdict):
    """Statistical class to store number metrics

    It has one local variable, *_value* accumulate things.
    It accept dict as input and the default value is always 0.
    But make sure every dict has the same keys, and the values
    support **+=** op.
    """
    def __init__(self):
        super().__init__(lambda: 0)

    def accumulate(self, val):
        assert isinstance(val, dict)
        assert 'count' not in val, 'keyword count is preserved in Metric'
        for k, v in val.items():
            self[k] += v
        self['count'] += 1

    def all_reduce(self):
        """Reduce the metrics across multiple workers

        It perform a distributed all_reduce.

        Returns:
            - reduced_value (float): the summed value
        """
        return {k: reduce_number(v) for k, v in self.items()}

    @property
    def step(self):
        return self._value['count']


class TimerResult(dict):
    """A nicely formatted dict for profiling results"""
    def __format__(self, fmt_str):
        str_list = ['Time']
        for name, value in self.items():
            item_str = f'{name} ' + format(value, fmt_str)
            str_list.append(item_str)
        return ' '.join(str_list)


class Timer(defaultdict):
    def __init__(self):
        super().__init__(lambda: TimeRecord(cuda=True))
        self.clear()

    def clear(self):
        builtin_keys = ('wall', 'io', 'gpu', 'loss', 'merge')
        # Ensure builtin keys are available
        for key in builtin_keys:
            self[key].reset()

    @property
    def last_result(self):
        return TimerResult(
            {name: time_record.last_interval for name, time_record in self.items()}
        )

    @property
    def total_result(self):
        return TimerResult(
            {name: time_record.total_interval for name, time_record in self.items()}
        )


class TimeRecord:
    """Record time interval

    Parameters:
        - cuda (bool): whether to do cuda synchronize when getting
            timestamps."""

    def __init__(self, cuda=False):
        self._start = 0  # record start time
        self._last_check_interval = 0  # the interval of last checkpoint
        self.total_interval = 0  # record time interval
        self.cuda = cuda

    def reset(self):
        self._start = 0
        self._last_check_interval = 0
        self.total_interval = 0

    def profile(self, generator):
        iterator = iter(generator)
        while True:
            try:
                self.start()
                value = next(iterator)
                self.checkpoint()
                yield value
            except StopIteration:
                break

    def __enter__(self):
        self.start()

    def __exit__(self, *_unused_args):
        self.checkpoint()

    def start(self):
        """Start the timer"""
        if self.cuda:
            torch.cuda.synchronize()
        self._start = time.time()

    def checkpoint(self):
        """End the timer"""
        if self.cuda:
            torch.cuda.synchronize()
        self.total_interval += time.time() - self._start

    @property
    def last_interval(self):
        last_interval = self.total_interval - self._last_check_interval
        self._last_check_interval = self.total_interval
        return last_interval
