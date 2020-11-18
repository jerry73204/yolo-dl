# YOLO Reproduction in Rust

## Prerequisites

- Rust toolchain
We suggest the [rustup](https://rustup.rs/) installer to install Rust toolchain.

- PyTorch 1.7.0
Install either by system package manager or by `pip3 install --user torch torchvision`.

## Usage

### Build

The command builds the entire project.


```sh
cargo build [--release]
```

The `--release` option in only needed in production. It gains about 5x performance, but it takes longer compilation time and the debug message is less verbose.

### Training

The training process is configured by [yolo-dl/train.json5](yolo-dl/train.json5). Run the command to start training,

```sh
cargo run --bin train --release
```

If it is configured correctly, it automatically starts from most recent checkpoint file.
