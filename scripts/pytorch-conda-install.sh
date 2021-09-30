#!/bin/bash
## Helper scripts to boostraping the developing environments on suzhou-cluster
## default use python3.7 cuda10.0, you can modified as you like
## needed to run on a GPU-available Host for warp-ctc and fsmn-kernel
## pu01 GPU driver version is too old, torch.cuda.is_available() will return False

dir=$1
python_version=3.7
cuda_version=10.0

if [ $# != 1 ]; then
    echo "Usage: $0 <dir>";
    exit 0;
fi

set -e
export CUDA_VISIBLE_DEVICES=0
mkdir -p $dir

echo "=========================================="
echo "    Create conda environment              "
echo "=========================================="

conda create -y -p $dir python=$python_version
source activate $dir

echo "=========================================="
echo "        Install PyTorch                   "
echo "=========================================="

# FIXME: pytorch-1.3.0 does not support cffi anymore, so warp-ctc and fsmn break
conda install -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=$cuda_version -c pytorch

echo "========================================="
echo "    Install other dependences            "
echo "========================================="

pip install editdistance filelock
conda install line_profiler
## too slow,  needed for mmi-loss
conda install -y pykaldi-cpu -c pykaldi

module load cmake cuda/$cuda_version gcc/7.3.0
export CC=`which gcc`
export CXX=`which g++`
export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"  # Cross-compile for Pascal, Volta and Turing

## wrap-ctc
cd $dir
git clone https://github.com/SeanNaren/warp-ctc.git

cd warp-ctc
mkdir -p build; cd build
cmake .. && make

cd ../pytorch_binding
python setup.py install

## fsmn (FIXME:需要换成 gitlab 的仓库)
cd $dir
git clone https://git.aispeech.com.cn/huangmingkun.sjtu/FSMN.git

cd fsmn_kernel
mkdir -p build; cd build
cmake .. && make

cd ../pytorch_binding
python setup.py install

## apex
git clone https://github.com/NVIDIA/apex
cp apex /tmp/apex -r && cd /tmp/apex  # lustre incompatible with pip
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
