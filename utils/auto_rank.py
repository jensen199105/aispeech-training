from pathlib import Path
from filelock import SoftFileLock

from . import slurm

def auto_rank(world_size, filename):
    """Automatically get the rank

    The rank will be determined by slurm job array id. If the job
    is not submitted by slurm, it falls back to file-based method.

    Args:
        world_size (int):
        filename (Path): the file location for file-based auto rank

    Return:
        rank (int): 0-based rank assigned for the current process
    """

    if slurm.is_slurm_job:
        rank = _slurm_auto_rank(world_size)
    else:
        print('Not submitted by slurm (No SLURM_JOB_ID), use file-based rank instead')
        rank = _file_auto_rank(world_size, filename)
    return rank


def _slurm_auto_rank(world_size):
    """Get the rank based on slurm job array id

    Args:
        world_size (int): the size of slurm job array should equal to world_size

    Returns:
        rank (int): 0-based rank assigned for the current process

    Raises:
        ValueError: if the job is not submitted by slurm
        RuntimeError: if the ``world_size`` != ``job_array_size``
    """
    # slurm task id is 1-based
    if slurm.world_size != world_size:
        raise RuntimeError(f'ERROR: Submitted {world_size}-GPUs job with array size {slurm.world_size}')
    return slurm.rank

def _file_auto_rank(world_size, filename):
    """Automatically get the rank based on a shared file

    The file-system should support flock / exclusive mode

    Args:
        world_size (int):
        filename (Path): the file location to communicate on

    Return:
        rank (int): 0-based rank assigned for the current process
    """

    assert isinstance(filename, Path)
    lock = SoftFileLock(str(filename.resolve()) + '.lock')
    with lock:
        # Get num workers which already get the rank
        try:
            with filename.open('r') as f:
                num_workers = int(f.readline())
        except FileNotFoundError:
            num_workers = 0

        # Increment by one if not the last rank, else delete it
        if num_workers + 1 < world_size:
            with filename.open('w') as f:
                f.write('{:n}\n'.format(num_workers + 1))
        else:
            try:
                filename.unlink()
            except FileNotFoundError:
                pass

    return num_workers
