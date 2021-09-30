import argparse
import yaml
from pathlib import Path

from ..utils import slurm
from ..utils.common import str2bool, str2dict
from .omega import parse_omega_args


def post_process_args(args):
    """Post process the arguments

    It performs the following actions:

    - When ``--resume`` is ``true``, we will load the args from checkpoint-dir and
    ignore all the command line args
    - Create the checkpoint-dir folder
    - Use yaml to save human-readable args for future loading
    """
    # Ensure the folder to store checkpoint exists
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    arg_file = ckpt_dir / 'args.yaml'
    if args.resume:
        # Use pyyaml to construct class automatically
        args = yaml.unsafe_load(arg_file.open())
        # Force the resume of previous run to be True
        args.resume = True
        # Disable model init
        args.init = args.kaldi_init = None
        # Assume data is split in previous failed run
        args.no_split = True
    else:
        yaml.dump(args, stream=arg_file.open('w'))

    return args


def get_conventional_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint-dir', type=Path, default='exp/test',
                        help='Folder to store checkpoint and log')
    # xkc09 added for data
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='The Base data directory, all the data including'
                        'feats, alignments and other locate in $DATA_DIR/tr and $DATA_DIR/cv')
    parser.add_argument('--feat', type=str2dict, default=[], action='append',
                        help='The training feature pipe as in Kaldi\n'
                             'e.g. ark: copy-feats scp:feats.scp ark:- |')
    parser.add_argument('--ali', type=str2dict, default=[], action='append',
                        help='The training alignment pipe as in Kaldi\n'
                             'e.g. ark: copy-align scp:ali.scp ark:- |')
    parser.add_argument('--lat', type=str2dict, default=[], action='append',
                        help='The training alignment rspec as in Kaldi\n'
                             'e.g. scp:lat.scp')
    parser.add_argument('--ivec', type=str2dict, default=[], action='append',
                        help='The training alignment pipe as in Kaldi\n'
                             'e.g. ark: copy-feats scp:ivec.scp ark:- |')
    # process config
    parser.add_argument('--stage', type=str, default=[], action='append', choices=['tr', 'cv'],
                        required=True, help='Specify the stages to launch')
    parser.add_argument('--use-sds', type=str2bool, default=False,
                        help='Use sds to accelerate training')
    parser.add_argument('--log-debug', type=str2bool, default=False,
                        help='Choose whether print debug info or not')
    # training config
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='The name of optimizer')
    parser.add_argument('--model', type=str2dict, required=True,
                        help='Neural network model parameters')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Initial base learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum of the optimizer, valid only for sgd')
    parser.add_argument('--nesterov', type=str2bool, default=False,
                        help='Nesterov of the optimizer, valid only for sgd')
    parser.add_argument('--scheduler', type=str2dict, default={'name': 'kaldi'},
                        help='Learning rate scheduler')
    parser.add_argument('--max-epoch', type=int, default=50,
                        help='The uppper bound of training epochs')
    parser.add_argument('--l2-factor', type=float, default=1e-2, dest='l2_factor',
                        help='L2 regularizer factor')
    parser.add_argument('--log-interval', type=int, default=360000,
                        help='Interval of logging batches')
    parser.add_argument('--dump-interval', type=int, default=10000,
                        help='Interval of dump intermediate result')
    parser.add_argument('--random-sweep', type=str2bool, default=False,
                        help='Enable random sweep to increase effective amount of training data')

    # FP16 mixed-precision training
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2', 'O3'], default='O0',
                        help='The apex.amp opt_level')

    # Model checkpointing
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument('--init', type=str,
                            help='Init by checkpoint saved in pytorch format')
    init_group.add_argument('--kaldi-init', type=str,
                            help='Init by kaldi nnet (must be converted to text form)')
    init_group.add_argument('--resume', type=str2bool, default=False,
                            help='Resume from last checkpoint')
    parser.add_argument('--max-keep', type=int, default=50,
                        help='Maximum number of saved parameters to keep')
    # multi-gpu
    parser.add_argument('--world-size', type=int, default=slurm.world_size,
                        help='Number of workers (gpus) in total, usually you dont need to specify on slurm')
    parser.add_argument('--merge-size', type=int, default=120000,
                        help='Number of frames to merge training model')
    parser.add_argument('--global-momentum', type=float, default=None,
                        help='Global momentum for BMUF, the model updates agressively for large momentum.'
                             ' The global momentum will be automatically set to an empirical value if -1 (by default)')
    parser.add_argument('--merge-function', type=str, default='BMUF-NBM', choices=['BMUF-NBM', 'BMUF-CBM', 'MA'],
                        help='Merge function for distributed training')
    parser.add_argument('--backend', type=str, default='nccl',
                        help='Bankend for multi gpu, DO NOT change unless you know what you are doing!')

    # Dynamically build loss and data loader
    parser.add_argument('--loss', type=str2dict, required=True,
                        help='Loss type and arguments',)
    parser.add_argument('--collector', type=str2dict, required=True,
                        help='How to make a data batch given multiple samples')

    # Split data
    parser.add_argument('--inplace-split', type=str2bool, default=False,
                        help='Split right in the source data directory')
    parser.add_argument('--no-split', type=str2bool, default=False,
                        help='Bypass the split stage for fast start-up')
    return parser


def parse_args() -> argparse.Namespace:
    """Parse the command line options into python arg object

    Ideally it accepts both omega style args and conventional CLI args
    """
    # TODO: deprecate conventional CLI parser
    parser = get_conventional_parser()
    omega_parser = argparse.ArgumentParser('Yaml configuration interface')
    omega_parser.add_argument(
        'overrides', nargs='*',
        help='Any key=value arguments to override config '
        'values (use dots for.nested=overrides)',
    )
    omega_parser.add_argument(
        '--conf', '-c', action='append',
        help='Yaml configuration files',
    )

    omega_cli_args, _ = omega_parser.parse_known_args()

    if omega_cli_args.conf:
        raw_args = parse_omega_args(omega_cli_args)
    else:
        raw_args = parser.parse_args()
    args = post_process_args(raw_args)

    return args
