import codecs
import argparse
from .utils import transform_pytorch2kaldi_nnet as transform_pytorch2kaldi_nnet

parser = argparse.ArgumentParser(description='')
parser.add_argument('--in-model', help='checkpoint-in-model')
parser.add_argument('--out-model', help='kaldi-out-model')
args = parser.parse_args()

kaldi_nnet = transform_pytorch2kaldi_nnet.trans_pytorch2kaldi(args.in_model)
file = codecs.open(args.out_model, 'w', 'utf-8')
file.write(kaldi_nnet)
file.close()
