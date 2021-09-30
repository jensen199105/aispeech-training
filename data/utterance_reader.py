import random
import logging
import itertools

from .reader import KaldiFeatReader, KaldiAliReader, KaldiLatticeReader, KaldiIvecReader, shared_offset


logger = logging.getLogger(__name__)


class UtteranceReader:
    """Load all the fields of one utterance from Kaldi rspec and pre-process

    A sample consists of (multiple) training features and (multiple)
    alignments.

    Attributes:
        sds_rate (int, optional): Percentage of the data used. Defaults to 1.

    Parameters:
        data_rspec: the data descriptor
        random_sweep (bool, optional): set True to randomly choose the starting frame offset from
                                       0 to `skip_frame - 1`.
        use_sds (bool): enable stochastic data sweeping

    data_rspec format:
        feat_rspecs (List[str]): list of feat rspecs, ususally viewed as
            feature
        ali_rspecs (List[str]): list of alignment rspecs, usually viewed
            as label
        ivec_rspecs (List[str]): list of i-vector rspecs
        lat_rspecs (List[str]): list of lattice rspecs

    Examples:
        >>> feat_rspecs = [{'rspec': 'ark:copy-feats scp:feat.scp ark:-|'}]  # REPLACE with a valid rspec
        >>> ali_rspecs = [{'rspec': 'scp:copy-int-vector scp:ali.scp ark:-|'}]  # REPLACE with a valid rspec
        >>> utt_reader = UtteranceReader([feat_rspecs, ali_rspesc])  # use the default parameters
        >>> for uid, fields in utt_reader:
        >>>    print(uid)
        >>>    print(fields)
    """

    def __init__(self, data_rspecs, random_sweep=False, use_sds=False):
        assert len(data_rspecs) == 4, 'Data rspecs insufficients, expected' \
            ' (feats_rspecs, ali_rspecs, ivec_rspecs or None, lat_rspecs or None)'
        self.data_rspecs = data_rspecs
        self.random_sweep = random_sweep
        self.use_sds = use_sds

        # FIXME: Use epoch based sds rate
        if use_sds:
            self.sds_rate = 0.5
        else:
            self.sds_rate = 1
        logger.info('SDS rate: {}'.format(self.sds_rate))

        feat_rspecs, ali_rspecs, ivec_rspecs, lat_rspecs = self.data_rspecs

        # Iterate over samples in all rspec files
        feat_list = [KaldiFeatReader(**feat) for feat in feat_rspecs]
        #ali_list = [KaldiAliReader(**ali) for ali in ali_rspecs]
        ali_list = [KaldiFeatReader(**ali) for ali in ali_rspecs]
        ivec_list = [KaldiIvecReader(**ivec) for ivec in ivec_rspecs]
        lat_list = [KaldiLatticeReader(**lat) for lat in lat_rspecs]
        self.readers_list = [feat_list, ali_list, ivec_list, lat_list]

    def __repr__(self):
        return f'{self.__class__.__name__} (random_sweep={self.random_sweep}, sds={self.use_sds})'

    def __iter__(self):
        """Read the feature and alignment into one sample

        Returns:
            fields (tuple): tuple containing (feat, ali, ivec, lattice)

                - feat (List[np.array]): features of the current sample
                - ali (List[np.array]): alignments of current sample
                - ivec (List[np.array]): list of i-vector of current sample, could
                  be empty list
                - lattice (List[kaldi.Lattice]): list of lattice objects defined in
                  pykaldi, could be empty list
        """

        # Iter through feat, ali, ivec, lat
        field_iter_list = []
        for field_readers in self.readers_list:
            field_iter_list.append(_combine_readers(field_readers))

        for sample in _combine_readers(field_iter_list):
            if self.random_sweep:
                shared_offset.update_offset()
            if random.random() <= self.sds_rate:
                yield sample


class FieldEmpty:
    """A special class to indicate empty generator"""


def _combine_readers(kaldi_readers):
    """

    Args:
        fields: list of kaldi***reader
    """

    if kaldi_readers:
        for samples in zip(*kaldi_readers):
            data_list = []
            utterance_id_set = set()
            for sample in samples:
                if isinstance(sample, FieldEmpty):
                    data_list.append([])
                else:
                    utterance_id, data = sample
                    utterance_id_set.add(utterance_id)
                    data_list.append(data)

            # Make sure they belong to the same utterence
            assert len(utterance_id_set) == 1, 'Utterence ID mismatch {}, please check if the scp files are all' \
                ' sorted in the same order: \n{}'.format(utterance_id_set, '\n'.join(
                    [str(rspec) for rspec in itertools.chain.from_iterable(kaldi_readers)]
                ))
            yield utterance_id_set.pop(), data_list
    # always yield a flag to indicate this field is empty
    else:
        yield from itertools.cycle([FieldEmpty()])
