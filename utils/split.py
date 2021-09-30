import re
import copy
import subprocess
import logging

import torch.distributed as dist

from .common import reduce_number

logger = logging.getLogger(__name__)


def split_data(data_dir, target_dir, stage, world_size):
    """Split the data and into target_dir

    It splits the data in ``data_dir/tr`` and ``data_dir/cv`` evenly into
    ``world_size`` shares, each worker will load its' own data based on the rank.
    """
    logger.debug(f'Splitting data {data_dir} -> {target_dir} into {world_size} chunks')
    script = 'dnn_split_naive.sh'
    real_data_dir = data_dir / stage
    target_data_dir = target_dir / stage
    cmd_list = [script, world_size, real_data_dir, target_data_dir]
    cmd_list = [str(arg) for arg in cmd_list]
    try:
        output = subprocess.check_output(cmd_list, shell=False)
        logger.info(output.decode().rstrip())
        fail_count = 0
    except subprocess.CalledProcessError as err:
        logger.fatal(err)
        fail_count = 1
    return fail_count


def resolve_data_dir(all_fields, split_dir):
    """Resolve the dir placeholder in templates into real data dir

    Args:
        template (list[list[str]]): template strings with ``SPLIT_PYRE`` as
                                    placeholder path
        split_dir (Path): the real data path to replace with

    Returns:
        rspecs (list[list[str]]): the rspecs with placeholder resolved to
                                  the real path.
    """
    resolved_fields = copy.deepcopy(all_fields)
    for fields in resolved_fields:
        for field in fields:
            field['rspec'] = re.sub(r'SPLIT_PYRE', str(split_dir), field['template'])
            del field['template']

    return resolved_fields


def get_per_worker_rspecs(all_field, data_dir, sdata_dir, stage, rank=0, world_size=1, no_split=False):
    """Get the tr&cv rspecs for current worker

    It splits all the files under ``data_dir`` into ``sdata_dir``, while maintaining
    the filename. `The` `SPLIT_PYRE` is going to be replaced by the real directory
    where the data of current worker locates in.

    Args:
        all_field (list[dict]): template strings with ``SPLIT_PYRE`` as
                                placeholder path. The list[str] should be
                                [feat_template, ali_template, ...]
        data_dir (Path): the original data path
        sdata_dir (Path): the root path to store *split data*, ``sdata`` is the
                          naming convention from Kaldi
        stage (str): the sub-directory (typically {'tr', 'cv'}) to split
        rank (int, optional): worker rank, 0 by default
        world_size (int, optional): world size, 1 by default
        no_split (bool, optional): whether to by-pass the split stage

    Returns:
        all_fields (tuple(feat, ali, ivec, lat)): field rspecs ready for use
    """
    for field in all_field:
        assert isinstance(field, list)  # for omegaconf ListConfig

    # The barrier is to warmup the distributed communication
    # The following barrier will raise file not found error without former one
    dist.barrier()
    if not no_split:
        logger.debug('Start to split data into {} splits, from {} to {}'.format(world_size, data_dir, sdata_dir))
        fail_count = 0
        if rank == 0:
            fail_count = split_data(data_dir, sdata_dir, stage, world_size)
        fail_count = reduce_number(fail_count)
        if fail_count > 0:
            logger.fatal(f'Error occurs during splitting {stage}. See rank0.log and slurm out for details')
            raise RuntimeError('Data split failed')
    dist.barrier()

    split = 'split{}/{}/'.format(world_size, rank)
    # Replace the "SPLIT_PYRE" in tr_feat and cv_feat string with `split`
    split_dir = sdata_dir / stage / split
    resolved_field = resolve_data_dir(all_field, split_dir)

    return resolved_field
