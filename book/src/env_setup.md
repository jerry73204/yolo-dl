# Environment Setup

## Prerequisites

In the rest of the book, we assume the Linux operating system with the following software requirements.

### Rust toolchain

We suggest the [rustup.rs](https://rustup.rs/) toolchain manager to install Rust toolchain.

### PyTorch 1.8.0, torchvision 0.8.1

This command installs the up-to-date PyTorch and torchvision.

```sh
pip3 install --user torch torchvision
```

If you prefer installing on Conda or on other platforms, visit [pytorch.org](https://pytorch.org/) to find appropriate commands.

### CUDA 10.2

Visit the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) find the installer for your platform.

### OpenCV 4.x

If you are using Ubuntu 20.04, OpenCV 4.2 can be installed from Ubuntu repository.

```sh
sudo apt install libopencv-dev
```

Ubuntu 18.04 or earlier version users must build OpenCV manually because official package is outdated. We suggest following the instructions in the [writeup](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/).

## Environment Variables

The environment variables in this section must set every time we start the program. It is suggested to put them in `.bashrc` or `.zshrc`.

The variables instruct the Rust program to emit verbose logs and backtrace for errors.

```sh
# (optional) show backtrace when error occurs
export RUST_BACKTRACE=1

# (optional) show logging message to INFO level
export RUST_LOG=info
```

The location of libtorch and ABI must be specified. This is an example setup if you install PyTorch by `pip3` on Ubuntu 18.04/20.04.

```sh
# (required) The libtorch directory
export LIBTORCH=$HOME/.local/lib/python3.8/site-packages/torch

# (required) The C++ ABI which libtorch is compiled with
export LIBTORCH_CXX11_ABI=0
```


## Development Environment

We suggest working on your favorite editor with [rust-analyzer](https://rust-analyzer.github.io/manual.html). It helps hunting common errors with ease.
