#!/bin/bash

# Copyright 2012-2015 Brno University of Technology (author: Karel Vesely), Daniel Povey
# Apache 2.0

# Begin configuration section.

stage=0 # stage=1 skips lattice generation
nj=4

nnet_forward=
acwt=0.10 # note: only really affects pruning (scoring is on lattices).
beam=13.0
lattice_beam=8.0
min_active=200
max_active=7000 # limit of active tokens
max_mem=50000000 # approx. limit to memory consumption during minimization in bytes
num_threads=1 # if >1, will use latgen-faster-parallel

skip_scoring=false
#scoring_opts="--min-acwt 4 --max-acwt 15"
scoring_opts=""
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the DNN and transition model is."
   echo "e.g.: $0 exp/dnn1/graph_tgpr data/test exp/dnn1/decode_tgpr"
   echo ""
   echo "This script works on plain or modified features (CMN,delta+delta-delta),"
   echo "which are then sent through feature-transform. It works out what type"
   echo "of features you used from content of srcdir."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo ""
   echo "  --nnet_forward <str>                             # DNN/models Forward specifier"
   echo "                                                   # from parent dir of <decode-dir>' (opt.)"
   echo ""
   echo "  --acwt <float>                                   # select acoustic scale for decoding"
   echo "  --scoring-opts <opts>                            # options forwarded to local/score.sh"
   echo "  --num-threads <N>                                # N>1: run multi-threaded decoder"
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
[ -z "$nnet_forward" ] && echo "No nnet forward specifier!" && exit 1;

mkdir -p $dir/log
echo $nj > $dir/num_jobs

# Possibly use multi-threaded decoder
thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

if [ $stage -le 0 ]; then
# Run the decoding in the queue,
  $decode_cmd --num-threads $((num_threads+1)) JOB=1:$nj $dir/log/decode.JOB.log \
    $nnet_forward \| \
    latgen-faster-ctc-mapped$thread_string --min-active=$min_active --max-active=$max_active \
      --max-mem=$max_mem --beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
      --allow-partial=true --word-symbol-table=$graphdir/words.txt \
      $graphdir/TLG.fst ark:- "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
fi

# Scoring
if [ $stage -le 1 ]; then
if ! $skip_scoring ; then
[ ! -x local/score_ctc.sh ] && echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
local/score_ctc.sh $scoring_opts --cmd "$local_cmd" $data $graphdir $dir || exit 1;
fi
fi

exit 0;
