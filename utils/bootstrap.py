import logging
import random
import socket
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from .. import __version__
from . import slurm
from .common import rm_file

logger = logging.getLogger(__name__)


def setup_root_logger(checkpoint_dir, rank, debug):
    """Setup the root logger for all the packages

    log lines with WARNING or higher level will be printed
    to stderr, while stderr will be collected as ERROR level
    """
    # log file prepare
    log_dir = checkpoint_dir / 'log'
    try:
        log_dir.mkdir(mode=0o755)
    except FileExistsError:
        pass
    log_file = log_dir / 'rank{}.log'.format(rank)

    basic_handler = logging.FileHandler(log_file)
    basic_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    handlers = [basic_handler]

    stdout_handler = logging.StreamHandler(sys.stderr)
    stdout_handler.setLevel(logging.WARNING)
    handlers.append(stdout_handler)

    if rank == 0:
        critical_handler = logging.FileHandler(log_dir / 'critical.log')
        critical_handler.setLevel(logging.WARNING)
        handlers.append(critical_handler)

    package_name = __name__.split('.')[0]
    root_logger = logging.getLogger(package_name)
    formatter = logging.Formatter(
        f'%(asctime)s[%(name)s]-%(levelname)s-%(message)s',
        datefmt='%H:%M:%S',
    )
    root_logger.setLevel(logging.DEBUG)
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    # Capture python's builtin warnings message
    logging.captureWarnings(True)
    redirect_stderr()

    logger.warning('Version {} >>> time {} >>> running on {} >>> slurm job id: {}'.format(
        __version__, time.asctime(), socket.gethostname(), slurm.job_id,
    ))


def redirect_stderr():
    """Redirect stderr to logger"""

    class LoggerWriter:
        """https://github.com/apache/airflow/pull/6767/files"""
        def __init__(self, target_logger, level=logging.INFO):
            self.logger = target_logger
            self.level = level

        def write(self, message):
            if message and not message.isspace():
                self.logger.log(self.level, message)

        def fileno(self):
            """
            Returns the stdout file descriptor 1.
            For compatibility reasons e.g python subprocess module stdout redirection.
            """
            return 1

        def flush(self):
            """MUST define flush method to exit gracefully"""

    sys.stderr = LoggerWriter(logger, logging.ERROR)


def determinize_random_state():
    """Initialize and setup configurations for reproducibility"""
    # force initialization of cuda context for reproducibility
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    # NOTE it's important to set random seed at the very beginning.
    random_seed = 777
    logger.debug(f'Seeding random state with {random_seed}')
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def init_distributed(backend, world_size, rank, checkpoint_dir):
    """Init the distributed communication group"""
    # multi-gpu initial
    logger.debug(f'Initializing {world_size} workers')
    # Remove the init file from previous version
    init_dir = checkpoint_dir / 'shared_distributed'
    if init_dir.is_file():
        rm_file(init_dir)

    init_dir.mkdir(parents=True, exist_ok=True)
    init_file = init_dir / f'slurm-{slurm.job_id}'
    init_method = init_file.resolve().as_uri()
    dist.init_process_group(backend, world_size=world_size, rank=rank, init_method=init_method)
    logger.debug('Init finished')
