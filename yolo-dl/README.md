# yolo-dl: YOLOv4 in Rust using tch

The project aims to reproduce [AlexeyAB's YOLOv4](https://github.com/AlexeyAB/darknet) from the ground up.
It tries to get the best parts of Rust, a programming language focused on perfomance and safety,
and expores good pratices to build machine learning toolchains in Rust.
It is still under development and is not feature complete.

The project provides these features:

- [par-stream](https://github.com/jerry73204/par-stream): Building block solution of parallel and asynchronous data pipeline, unlike PyTorch's all-in-one `DataParallel`.
- [tfrecord-rust](https://github.com/jerry73204/rust-tfrecord): Fluent integration with TensorBoard and `*.tfrecord` files.
- Integrity guaranteed model and experiment configuration ser/deserialization, powered by [serde](https://github.com/serde-rs/serde).
- Type-checked scientific computation. Think of the `BBox` type, which length units are marked by runtime cost-free `PixelUnit` and `RatioUnit` markers.

## Usage

Before getting started, download COCO 2017 dataset and modify the config file `train.json5`.
Run the training program and good luck.

```sh
cargo run --release --bin train
```

## License

MIT license. See [LICENSE file](LICENSE.txt).
