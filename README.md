# YOLO Reproduction in Rust

The project aims to reproduce [AlexeyAB's YOLOv4](https://github.com/AlexeyAB/darknet).
It gets the best parts of Rust to build fast, parallel and concurrent data pipeline.
It helps us learn good practices to build machine learning machinery in Rust.

It is still under development and is not feature complete.

## Get Started

### Dependencies

To get started, install following required toolchains and dependencies.

- Rust toolchain
We suggest the [rustup](https://rustup.rs/) installer to install Rust toolchain.

- PyTorch 1.7.0
Install either by system package manager or by `pip3 install --user torch torchvision`.

- CUDA 11.x

### Development Environment

The environment variables enable verbose messaging from program.

```sh
export RUST_BACKTRACE=1  // show backtrace when panic
export RUST_LOG=info     // verbose logging
```

We suggest working on your favorite editor with [rust-analyzer](https://rust-analyzer.github.io/manual.html). It helps hunting common errors with ease.

## Usage

### Build

The command builds the entire project.

```sh
cargo build [--release]
```

The `--release` option in only needed in production. It gains about 5x performance, but it takes longer compilation time and produces less verbose debug message.

### Training

The training process is configured by [yolo-dl/train.json5](yolo-dl/train.json5). Run the command to start training,

```sh
cargo run --bin train --release
```

If it is configured correctly, it automatically starts from most recent checkpoint file.

## Related Projects

- [tch-rs](https://github.com/LaurentMazare/tch-rs)
Rust bindings to libtorch library.

- [serde](https://github.com/serde-rs/serde)
The declarative data ser/deserialization library enables us to write complex but comprehensive configuration files.

- [par-stream](https://github.com/jerry73204/par-stream)
Building blocks of parallel and asynchronous data pipeline, enabling more fine-grand control than PyTorch's `DataParallel`.

- [tfrecord-rust](https://github.com/jerry73204/rust-tfrecord)
Fluent integration with TensorBoard and `*.tfrecord` files.

- [voc-dataset-rs](https://github.com/jerry73204/voc-dataset-rs)
PASCAL Visual Object Classes (VOC) dataset toolkit.

- [coco-rs](https://github.com/jerry73204/coco-rs)
COCO dataset toolkit.

## License

MIT license. See [LICENSE file](LICENSE.txt).
