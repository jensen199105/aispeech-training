from dataclasses import dataclass, field
from typing import List, Optional, Any

from omegaconf import MISSING

from ..trainer.optimizer import BMethod
from ..utils import slurm


@dataclass
class Data:
    data_dir: str = MISSING
    feat: List[Any] = MISSING
    ali: Optional[List[Any]] = field(default_factory=list)
    lat: Optional[List[Any]] = field(default_factory=list)
    ivec: Optional[List[Any]] = field(default_factory=list)
    inplace_split: bool = False
    no_split: bool = False
    use_sds: Optional[bool] = False
    random_sweep: Optional[bool] = False


@dataclass
class Distributed:
    world_size: int = slurm.world_size
    merge_size: int = 120000
    global_momentum: Optional[float] = None
    merge_function: BMethod = BMethod.BMUF_NBM
    backend: str = 'nccl'


@dataclass
class Optimizer:
    optimizer: str = 'sgd'
    lr: float = 1e-5
    momentum: float = 0.9
    l2_factor: float = 1e-2
    nesterov: bool = False


@dataclass
class Trainer:
    checkpoint_dir: str = MISSING
    stage: List[str] = ('tr', 'cv')
    log_debug: bool = False
    max_epoch: int = 50
    log_interval: int = 360000
    dump_interval: int = 10000
    max_keep: int = 50
    amp_opt_level: str = 'O0'  # Choices are 'O0', 'O1, 'O2', 'O3'


@dataclass
class ASRConfig:
    hparams: Optional[Any] = None
    data: Data = Data()
    model: Any = MISSING
    scheduler: Any = field(default_factory=lambda: {'name': 'kaldi'})
    loss: Any = MISSING
    collector: Any = MISSING
    optim: Optimizer = Optimizer()
    dist: Distributed = Distributed()
    trainer: Trainer = Trainer()
    init: Optional[str] = None
    kaldi_init: Optional[str] = None
    resume: bool = False
