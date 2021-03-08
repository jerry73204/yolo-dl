# About yolo-dl Project

## Overview

The project aims to reproduce [AlexeyAB's YOLOv4](https://github.com/AlexeyAB/darknet).
It gets the best parts of Rust to build fast and concurrent data pipeline.
It helps us learn good practices to build machine learning applications in Rust.

The project is built atop of [tch-rs](https://github.com/LaurentMazare/tch-rs), a Rust binding to libtorch. The libtorch is the C++ backend of PyTorch. Our project shares most common tensor operations and automatic differentiation engine with PyTorch.

Our project benefits from Rust's memory safety and fearless concurrency features ([article](https://doc.rust-lang.org/book/ch16-00-concurrency.html)). It enables us to build efficient and highly customization data pipeline without tears. Rust,as a first class citizen in low level programming, can link to C/C++ libraries and CUDA kernels directly. It helps adopting existing implementations with ease.

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

  Taiwanese road scenes and object images and labels, maintained by [Institute for Information Industry (III)](https://web.iii.org.tw/) in Taiwan.

## License

The project is licensed under MIT.

If you are using the Formosa dataset feature, please cite the follow text in your preferred language.

```
(中文)
Formosa自駕深度學習資料庫, 財團法人資訊工業策進會

(English)
The Formosa Dataset, created by Institute for Information Industry
```
