# About yolo-dl Project

The project aims to reproduce [AlexeyAB's YOLOv4](https://github.com/AlexeyAB/darknet).
It gets the best parts of Rust to build fast and concurrent data pipeline.
It helps us learn good practices to build machine learning applications in Rust.

The project is built atop of [tch-rs](https://github.com/LaurentMazare/tch-rs), a Rust binding to libtorch. The libtorch is the C++ backend of PyTorch. Thus our project shares most common tensor operations and automatic differentiation engine with PyTorch.

Our project benefits from Rust's memory safety and fearless concurrency features ([article](https://doc.rust-lang.org/book/ch16-00-concurrency.html)). It enables us to build efficient and highly customization data pipeline without tears. Rust,as a first class citizen in low level programming language, can  biliterate with C libraries and CUDA kernels directory. It helps adopting existing implementations with ease.
