"""Field readers for kaldi format feat, alignment, ivector and lattice."""
import math
import random
import logging

import numpy as np

from . import kaldi_io


logger = logging.getLogger(__name__)


class SharedOffset:
    """A dummy container class to hold the offset and share among different readers"""

    def __init__(self):
        self.offset = 0
        self.random_generator = random.Random(777)

    def update_offset(self):
        """A trivial method to update the offset to a random value"""
        self.offset = self.random_generator.random()


# We need the dummy instance for inplace modification in function closure
shared_offset = SharedOffset()


def feat_delay(feat, target_delay=0):
    """Pad feature by last frame to the same length of delayed alignment"""
    return np.pad(feat, [(0, target_delay), (0, 0)], mode='edge')


def ali_delay(ali, target_delay, pad_value=-1):
    r"""Pad alignment with 0 for target delay.

    Args:
        ali: original alignment.
        target_delay: delay original alignment with #target_delay frames.
        pad_value: should be ce loss' ignore_index, default -1.
    """
    return np.pad(ali, (target_delay, 0), mode='constant', constant_values=pad_value)


def feature_splice(xs, splice):
    """Do frame expansion for longer context"""
    left, right = splice
    assert left >= 0 and right >= 0, 'The splice {} is invalid'.format(splice)
    if left == 0 and right == 0:
        return xs

    padded_xs = np.pad(xs, [(left, right), (0, 0)], mode='edge')

    def sliding_window(a, window, step_size):
        """Reshape a numpy array 'a' of shape (n, x) to form shape((n - window_size + 1), window_size, x))"""
        shape = a.shape[:-1] + (a.shape[-1] - window + 2 - step_size, window)
        shape = (a.shape[0] - window + 2 - step_size, window) + a.shape[1:]
        strides = (a.strides[0] * step_size,) + a.strides
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    window_ys = sliding_window(padded_xs, (left + right + 1), 1)
    flatten_ys = window_ys.reshape(xs.shape[0], -1)
    return flatten_ys


def stack(xs, stack_frame):
    """Stack multiple frames into one single frame

    Args:
        xs (np.array): input feature matrix
        stack_frame (int): #frames to stack

    Returns:
        ys (np.array)
    """
    assert stack_frame > 0
    length = len(xs) - len(xs) % stack_frame
    ys = xs[:length].reshape(xs.shape[0] // stack_frame, -1)
    return ys


def apply_skip_frame(xs, skip_frame):
    """Apply skip for both feature and alignment"""
    assert skip_frame > 0, 'either skip-frame or skip-inner <= 0'
    offset = math.floor(shared_offset.offset * skip_frame) % skip_frame
    xs = xs[offset::skip_frame]
    return xs


class KaldiFeatReader:
    """Read the feature from kaldi

    .. warning::

        The global transform file format differs from kaldi's, use
        with caution.

    Args:
        rspec (str): the feature rspec string
        transform (str or None): the global transform matrix file path, first row
                                 `Addshift`, second row `Rescale`. No ``[`` or ``]`` is needed.
        splice (tuple(int, int)): context length for previous and future frames
        stack_frame (int, optional): Google style stacking+decimation,
        skip_frame (int, optional): number of frames to skip out-side the model
                                    it is used to skip the label if skip_label is True
        target_delay (int, optional): delay the label to utilize future frames,
                                      it's applied after `skip_frame`
    """

    def __init__(self, rspec, transform=None, splice=(0, 0), stack_frame=1, skip_frame=1, target_delay=0):
        self.feat_rspec = rspec
        self.splice = splice
        self.stack_frame = stack_frame
        self.skip_frame = skip_frame
        self.target_delay = target_delay
        if transform:
            logger.info(f'Use feature transform from {transform}')
            self.transform = np.recfromtxt(transform).astype('float32')
        else:
            self.transform = None

    def __iter__(self):
        """Read the feature rspec and post-process

        Yield:
            utterence_id (str): the utterence ID of current sample
            feat (np.array): the feature vector of current sample
        """

        for utterence_id, feat in kaldi_io.read_mat_ark(self.feat_rspec):
            if self.transform is not None:
                feat = (feat + self.transform[0]) * self.transform[1]
            feat = feature_splice(feat, self.splice)
            # TODO: stack option not validated
            feat = stack(feat, self.stack_frame)
            feat = apply_skip_frame(feat, self.skip_frame)
            feat = feat_delay(feat, self.target_delay)
            yield utterence_id, feat


class KaldiAliReader:
    """Read the alignment from kaldi

    Args:
        rspec (str): the alignment rspec string, e.g. ``scp:ali.scp``
        target_delay (int, optional): delay the label to utilize future frames,
                                      it's applied after ``skip_frame``
        skip_frame (int, optional): number of frames to skip out-side the model
                                    it is used to skip the alignment here
    """

    def __init__(self, rspec, skip_frame=1, target_delay=0):
        self.ali_rspec = rspec
        self.skip_frame = skip_frame
        self.target_delay = target_delay

    def __iter__(self):
        """Read the alignment rspec and post-process

        Yields:
            utterence_id (str): the utterence ID of current sample
            ali (np.array): the alignment of current sample
        """
        # the terminology is the same for `feature`
        for utterence_id, ali in kaldi_io.read_vec_int_ark(self.ali_rspec):
            ali = apply_skip_frame(ali, self.skip_frame)
            ali = ali_delay(ali, self.target_delay)
            # Use long int for pytorch's label requirements
            yield utterence_id, ali.astype(np.int64)


class KaldiLatticeReader:
    """Read the lattice from kaldi

    .. warning::

        Since we load data from disk in a background process, we have to serialize the lattice
        into byte-stream and then deserialize it. It causes memory leaking due to pykaldi's implementation,
        check this issue https://github.com/pykaldi/pykaldi/issues/176.

    Args:
        rspec (str): the lattice rspec string, e.g. ``scp:lat.scp``
    """

    def __init__(self, rspec):
        self.lat_rspec = rspec

    def __iter__(self):
        """Read the lattice rspec using pykaldi

        Args:
            rspec (str): the lattice rspec string

        Yield:
            utterence_id (str): the utterence ID of current sample
            lattice (kaldi.fstext.LatticeVectorFst): the lattice object of current sample
        """
        # Lazy init pykaldi because it's only used to read lattice
        from kaldi.util.table import SequentialLatticeReader
        # Inform the consumer to restart reading the rspec
        restart = True
        for utterence_id, _ in SequentialLatticeReader(self.lat_rspec):
            if restart:
                lat = self.lat_rspec
                restart = False
            else:
                lat = None
            # The naive Lattice is not pickable, we have to serialize it manually
            # Temp work-around
            yield utterence_id, lat


class KaldiIvecReader:
    """Read the i-vector from kaldi
    Not implemented yet!
    """

    def __init__(self, rspec):
        self.ivec_rspec = rspec

    def __iter__(self):
        """Read the ivector rspecs using pykaldi

        Yields:
            utterence_id (str): the utterence ID of current sample
                                ivec
        """
        # TODO: support ivector reading
        raise NotImplementedError('I-vector features are not supported yet')
