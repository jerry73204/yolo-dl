# darknet-config

It provides utilities to work with AlexeyAB's [darknet](https://github.com/AlexeyAB/darknet) configuration and weights files, written in Rust.
It features [serde](https://crates.io/crates/serde)-compatible configuration, weights file loading and safe model types.

## Usage

### Show Layer Information

To print the layers and shapes of a configuration file,

```sh
cargo run --bin darknet-config info yolov4.cfg
```

### Plot Computation Graph

To plot the computation graph of the configuration file,

```sh
cargo run --bin darknet-config \
    make-dot-file \
    yolov4.cfg \
    output.dot
```

Then, convert the DOT file to SVG file. You can change the `-Tsvg` option to `-Tpng` to export a PNG image.

```sh
dot -Tsvg output.dot > output.svg
```

## License

MIT license. See [LICENSE file](LICENSE.txt).
