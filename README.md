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

## Documentation

### Book

The book can be found in [book](book) directory. It covers the environment setup and training configurations. To read the book,

- read it directly on GitHub [here](book/src/SUMMARY.md), or
- read the included [README](book/README.md) to find instructions to read it locally.

### API Documentation

Most documenation are inline in the code for now.

Compile the documenation and open it in browser. Please search for `yolo-dl-doc` crate to see the full documenation.

```sh
cargo doc --open
```

If you are developing on the remote site, you can open a remotedocument server with `cargo-docserve`. Please search for `yolo-dl-doc` crate in your browser.

```sh
# install if you run cargo-docserve for the first time
cargo install cargo-docserve

# run doc server
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

- [iii-formosa-dataset](https://docs.rs/iii-formosa-dataset)
The Institute for Information Industry Formosa dataset toolkit.

- The [Institute for Information Industry Formosa dataset](https://www.iii.org.tw/Product/TransferDBDetail.aspx?tdp_sqno=3345&fm_sqno=23)
Taiwanese road scenes and object labels.

## License

MIT license. See [LICENSE file](LICENSE.txt).
