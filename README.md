# YOLO Reproduction in Rust

The project aims to reproduce [AlexeyAB's YOLOv4](https://github.com/AlexeyAB/darknet).
It gets the best parts of Rust to build fast, parallel and concurrent data pipeline.
It helps us learn good practices to build machine learning machinery in Rust.

It is still under development and is not feature complete.

## Projects

- **darknet-config**: Parsing AlexeyAB's Darknet configuration file.
- **model-config**: The JSON5 based architecture description format used by this project.
- **tch-goodies**: Extension to [tch](https://github.com/LaurentMazare/tch-rs) crate.
- **yolo-dl**: The building blocks of YOLO detection model and preprocessors.
- **train**: The multi-GPU training program.

## Environment Setup

### Dependencies

To get started, install following required toolchains and dependencies.

- Rust toolchain
We suggest the [rustup](https://rustup.rs/) installer to install Rust toolchain.

- PyTorch 1.7.0
Install either by system package manager or by `pip3 install --user torch torchvision`.

- CUDA 10.x or 11.x

### Environment Variables

The section presents manidatory and optional environment variables. You may put the setup in your `.bashrc` or `.zshrc`.

We suggest the setup for Rust toolchain. It instructs the program to show backtrace when error occurs and shows more verbosed messages.

```sh
# (optional) show backtrace when error occurs
export RUST_BACKTRACE=1

# (optional) show logging message to INFO level
export RUST_LOG=info
```

The directory of libtorch library and its ABI must be respectively specified. This is an example setup if you installs PyTorch by Python 3 `pip` on Ubuntu 18.04/20.04.

```sh
# (required) libtorch library directory
export LIBTORCH=$HOME/.local/lib/python3.8/site-packages/torch

# (required) The C++ ABI which libtorch is compiled with
export LIBTORCH_CXX11_ABI=0
```

### Development Environment

We suggest working on your favorite editor with [rust-analyzer](https://rust-analyzer.github.io/manual.html). It helps hunting common errors with ease.

## Build the Project

The command builds the entire project.

```sh
cargo build [--release]
```

The `--release` option in only needed in production. It gains about 5x performance, but takes longer compilation time and produces less verbose debug message.

## Train a Model

### Run Training

The command trains a model. It loads `train.json5` configuration file in current directory. You can specify custom configuration file with `--config /path/to/config.json5`.

```sh
# Start training
cargo run --release --bin train -- [--config train.json5]
```

### Profile the Training

The `profiling` Cargo feature allows you to profile the timing of the data pipeline. It is useful to investigate the performance bottleneck.

```sh
cargo run --release --bin train --features profiling
```

It shows available timing profiles on terminal.

```
Feb 17 19:32:02.184  INFO yolo_dl::profiling:registered timing profile 'pipeline'
Feb 17 19:32:02.185  INFO yolo_dl::profiling:registered timing profile 'cache loader'
```

You can set `YOLODL_PROFILING_WHITELIST` environment varianble to show specific timing profile.

```sh
env YOLODL_PROFILING_WHITELIST='pipeline' \
    cargo run --release --bin train --features profiling
```

### Tuning Training Performance

The default Rust toolchain has debug and release profiles. You can add `--release` option to cargo commands to switch from debug to release mode. For example,

```sh
cargo run [--release]
cargo test [--release]
```

The default release profile settings can be found in `Cargo.toml`. It enables verbose debug messages and debug assertions.

```toml
[profile.release]
debug = true
debug-assertions = true
overflow-checks = true
lto = false
```

To get the best runtime performance, you can disable debugging features. Use it with caution that it disables several numeral checks such as NaN detection.

```toml
[profile.release]
debug = false
debug-assertions = false
overflow-checks = false
lto = true
```

More profile options can be found in Cargo reference.

https://doc.rust-lang.org/cargo/reference/profiles.html

### Show Statistics in TensorBoard

Set the logging diretory in `train.json` configuration file.

```json
"logging": {
    "dir": "logs-coco",
    // ...
}

```

After your training program is started, you can open a TensorBoard server to read the saved statistics.

```sh
tensorboard --bind_all --logdir logs-coco/
```

## Model Configuration

### Inspect a NEWSLABv1 Model Configuration File

NEWSLABv1 is a model description format used by NEWSLAB. To inspect a configuration file `model-config/tests/cfg/yolov4-csp.json5` for example,

```sh
cargo run --release --bin model-config -- info model-config/tests/cfg/yolov4-csp.json5
```

It can plot the model architecture into an image. For export an SVG file,


```sh
// export GraphViz DOT file
cargo run --release --bin model-config -- \
    make-dot-file \
    model-config/tests/cfg/yolov4-csp.json5 \
    image.dot

// convert DOT to SVG
dot -Tsvg image.dot > image.svg
```

### Inspect a Darknet Configuration File

The `darknet-config` is a toolkit to inspect configuration files from darknet project ([repo](https://github.com/AlexeyAB/darknet)). To show the model information,

```sh
cargo run --release --bin darknet-config -- info yolov4.cfg
```

The `make-dot-file` subcommand can plot the computation graph.

```sh
// export GraphViz DOT file
cargo run --release --bin darknet-config -- \
    make-dot-file yolov4.cfg image.dot

// convert DOT to SVG
dot -Tsvg image.dot > image.svg
```

## Documentation

### API Documentation

Most documenation are inline in the code for now.

Compile the documenation and open it in browser. Please search for `yolo-dl-doc` crate to see the full documenation.

```sh
cargo doc --open
```

If you are developing on the remote site, you can open a remotedocument server with `cargo-docserve`. Please search for `yolo-dl-doc` crate in your browser.

```sh
// install if you run cargo-docserve for the first time
cargo install cargo-docserve

// run doc server
cargo docserve
```

## Related Projects

- [tch-rs](https://github.com/LaurentMazare/tch-rs)
Rust bindings to libtorch library.

- [serde](https://github.com/serde-rs/serde)
The declarative data ser/deserialization library enables us to write complex but comprehensive configuration files.

- [tch-serde](https://github.com/jerry73204/tch-serde)
De/Serialization of data types in tch-rs.

- [par-stream](https://github.com/jerry73204/par-stream)
Building blocks of parallel and asynchronous data pipeline, enabling more fine-grand control than PyTorch's `DataParallel`.

- [tfrecord-rust](https://github.com/jerry73204/rust-tfrecord)
Fluent integration with TensorBoard and `*.tfrecord` files.

- [voc-dataset-rs](https://github.com/jerry73204/voc-dataset-rs)
PASCAL Visual Object Classes (VOC) dataset toolkit.

- [coco-rs](https://github.com/jerry73204/coco-rs)
COCO dataset toolkit.

- [Formosa dataset](https://www.iii.org.tw/Product/TransferDBDetail.aspx?tdp_sqno=3345&fm_sqno=23) from Institute for Information Industry in Taiwan which the project mainly works on.

## License

MIT license. See [LICENSE file](LICENSE.txt).
