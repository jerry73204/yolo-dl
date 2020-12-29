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

The environment variables enable verbose messaging from program. You can copy them to your `.bashrc` or `.zshrc`.

```sh
export RUST_BACKTRACE=1  // show backtrace when panic
export RUST_LOG=info     // verbose logging
```

We suggest working on your favorite editor with [rust-analyzer](https://rust-analyzer.github.io/manual.html). It helps hunting common errors with ease.

## Build the Project

### Prerequisites

The project requires PyTorch and CUDA 10.x/11.x to be installed.

### Build

The command builds the entire project.

```sh
cargo build [--release]
```

The `--release` option in only needed in production. It gains about 5x performance, but takes longer compilation time and produces less verbose debug message.

## Usage

### Training

The command trains a model. The `--config train.json5` is needed only when you wish to specify your custom configuration file.

```sh
cargo run --release --bin train -- [--config train.json5]
```

### Profile the Training

The `profiling` feature allows you to profile the timing of the data pipeline. It is useful to determine the performance bottleneck.

```sh
cargo run --release --bin train --features profiling --manifest-path train/Cargo.toml
```

### Inspect a Darknet Configuration File

The `darknet-config` is a toolkit to inspect configuration files from darknet project ([repo](https://github.com/AlexeyAB/darknet)). To show the model information,

```sh
cargo run --release --bin darknet-config -- info yolov4.cfg
```

The `make-dot-file` subcommand can plot the computation graph in DOT format. For example, to save the plot to `output.dot`,

```sh
cargo run --release --bin darknet-config -- make-dot-file yolov4.cfg output.dot
```

## Documentation

You can generate the API documentation from the source code. To open the doc in browser,

```sh
cargo doc --open
```

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
