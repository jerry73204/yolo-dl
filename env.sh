#!/usr/bin/env bash
PYTHON_VERSION=3.8
CUDA_VERSION=11.1

script_dir="$(dirname $(realpath $0))"
venv_dir="$script_dir/venv"

# pytorch
export LD_LIBRARY_PATH="${venv_dir}/lib/python${PYTHON_VERSION}/site-packages/torch/lib:$LD_LIBRARY_PATH"
export LIBTORCH="${venv_dir}/lib/python${PYTHON_VERSION}/site-packages/torch"
export LIBTORCH_CXX11_ABI=0

# cuda
export PATH="/usr/local/cuda-${CUDA_VERSION}/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64:$LIBRARY_PATH"

virtualenv "${venv_dir}"
source venv/bin/activate
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
