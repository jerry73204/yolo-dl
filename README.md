# darknet-config

It provides utilities to work with AlexeyAB's [darknet](https://github.com/AlexeyAB/darknet) configuration and weights files, written in Rust.
It features [serde](https://crates.io/crates/serde)-compatible configuration, weights file loading and safe model types.

## Usage

To print the summary of YOLOv4 configuration and weights,

```sh
cargo run --example info yolov4.cfg yolov4.weights
```

## License

MIT license. See [LICENSE file](LICENSE.txt).
