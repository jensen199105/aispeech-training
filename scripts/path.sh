#!/bin/bash
# This path should be placed in ASR_ROOT/asr/scripts to use out-of-the-box
TOOL_DIR=/mnt/lustre/aifs/gt/gtdata/tools
module purge
module add gcc/7.3.0
module add cuda/10.1
module add cudnn/7.6.1-cuda10.0
module add imkl/2017.3.196
module add mpich/ge/gcc/64/3.2

module add anaconda/3
source deactivate

# NOTICE: Get the location of scripts dir, under the assumption that this file is
# located at ASR_ROOT/asr/scripts/
PYTORCH_ASR_ROOT=$TOOL_DIR/pytorch-asr
SCRIPT_DIR=$PYTORCH_ASR_ROOT/asr/scripts
CONDA_ENV=$TOOL_DIR/asr-nightly
source activate $CONDA_ENV
export KALDI_ROOT=$TOOL_DIR/kaldi_aispeech_2080ti

# kaldi 包
# export KALDI_ROOT=/mnt/lustre/aifs/home/hl302/tools/KALDI/kaldi_aispeech
export PATH=$PWD:$PWD/local:$PWD/utils:$SCRIPT_DIR:$SCRIPT_DIR/local:$SCRIPT_DIR/utils/:$KALDI_ROOT/src/lmbin:$KALDI_ROOT/src/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/nnet0bin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/onlinebin/:$KALDI_ROOT/src/ivectorbin/:$PATH
export LC_ALL=C

MAX_THREADS=3
export MKL_NUM_THREADS=$MAX_THREADS
export NUMEXPR_NUM_THREADS=$MAX_THREADS
export OMP_NUM_THREADS=$MAX_THREADS
export OPENBLAS_NUM_THREADS=$MAX_THREADS
export VECLIB_MAXIMUM_THREADS=$MAX_THREADS

export PATH=$KALDI_ROOT/tools/openfst/bin:$PATH
export LD_LIBRARY_PATH=$KALDI_ROOT/tools/openfst-1.6.1/lib:$LD_LIBRARY_PATH

# pytorch asr 包
export PYTHONPATH=$PYTORCH_ASR_ROOT:$PYTHONPATH

export NCCL_DEBUG=VERSION

# 更详细的 NCCL debug 信息
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
