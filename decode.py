"""Entrypoint of asr model inference, it outputs the kaldi format posterior to stdout"""
import sys
import argparse

import torch

from .trainer.checkpoint import Checkpoint
from .data.kaldi_io import write_mat
from .data.batch_loader import BatchLoader
from .data.utterance_reader import UtteranceReader
from .data.collector import SequenceCollector
from .utils.common import read_log_prior


def get_parsed_args():
    """Parse arguments for decoding"""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dir', type=str, required=True, help='checkpoint-dir')
    parser.add_argument('--feat-rspec', type=str, required=True, help='feat rsepecifier')
    parser.add_argument('--transform', type=str, help='feature transform')
    parser.add_argument('--skip-frame', type=int, default=1, help='skip frame')
    parser.add_argument('--target-delay', type=int, default=0, help='target delay')
    parser.add_argument('--temperature', type=float, default=1, help='Softmax temperature')

    parser.add_argument('--prior', type=str, default=None, help='state prior')
    parser.add_argument('--prior-scale', type=float, default=1.0, help='state prior scale')

    parser.add_argument('--stream-size', type=int, default=80, help='Max #sentences per batch')
    parser.add_argument('--frame-limit', type=int, default=4096, help='Max #frames per batch')
    args = parser.parse_args()
    return args


def decode(model, data_queue, log_prior, temperature=1):
    """Run decode

    Read in data batch and writes posterior to stdout.

    Args:
        - model (asr.model.Model): well-trained model
        - data_queue (Iterable[asr.data.Batch]): queue of data batch
        - log_prior (np.array): the class prior distribution which will be
            subtracted on the final posterior.
        - temperature (float): softmax temperature, high temperature gives
            uniform distribution.
    """
    model.eval()
    with torch.no_grad():
        for batch in data_queue:
            key = batch['uid']
            output = model.decode(batch)
            xlen = output.length
            ys = output.tensor
            ys = torch.nn.functional.log_softmax(ys / temperature, dim=2).cpu().numpy()
            # B, T, C
            # ys = np.transpose(ys, (1, 0, 2))

            for uid, prob, length in zip(key, ys, xlen):
                write_mat(sys.stdout.buffer, prob[:int(length)] - log_prior, uid)


def main():
    """Main entrypoint

    You should call this at your own decode.py after `add_model`
    """
    args = get_parsed_args()
    use_gpu = torch.cuda.is_available()

    model = Checkpoint.load_model_from_dir(args.dir)
    if use_gpu:
        model.cuda()

    # load prior
    if args.prior:
        log_prior = args.prior_scale * read_log_prior(args.prior)
    else:
        log_prior = 0

    feat = {
        'rspec': args.feat_rspec,
        'transform': args.transform,
        'skip_frame': args.skip_frame,
        'target_delay': args.target_delay
    }

    data_rspecs = [[feat], [], [], []]

    collector = SequenceCollector(args.stream_size, args.frame_limit)
    utt_reader = UtteranceReader(data_rspecs, random_sweep=False)
    data_queue = BatchLoader(utt_reader, collector)

    decode(model, data_queue, log_prior, args.temperature)


if __name__ == '__main__':
    main()
