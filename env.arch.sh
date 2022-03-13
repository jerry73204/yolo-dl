#!/usr/bin/env bash
PYTHON_VERSION=3.8
CUDA_VERSION=11.1

script_dir="$(dirname $(realpath $0))"

wget -nc 'https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.9.1%2Bcu111.zip'
unzip -n libtorch-cxx11-abi-shared-with-deps-1.9.1+cu111.zip

# libtorch
export CC=gcc-8
export CXX=g++-8
export LD_LIBRARY_PATH="${script_dir}/libtorch/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="${script_dir}/libtorch/lib:$LIBRARY_PATH"
export LIBTORCH="${script_dir}/libtorch"
export LIBTORCH_CXX11_ABI=1

# cuda
export PATH="/opt/cuda-${CUDA_VERSION}/bin:$PATH"
export LD_LIBRARY_PATH="/opt/cuda-${CUDA_VERSION}/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="/opt/cuda-${CUDA_VERSION}/lib64:$LIBRARY_PATH"

# virtualenv3 --python "/usr/bin/python${PYTHON_VERSION}" "${venv_dir}"
# source venv/bin/activate
# pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
