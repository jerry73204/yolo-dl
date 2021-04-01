//! The crate provides extension to [Tensor](tch::Tensor) type and utilities.

pub mod activation;
pub mod bbox;
mod common;
pub mod compound_tensor;
pub mod detection;
pub mod ratio;
pub mod size;
pub mod tensor;
pub mod unit;
mod utils;
pub mod module;

pub use activation::*;
pub use bbox::*;
pub use compound_tensor::*;
pub use detection::*;
pub use ratio::*;
pub use size::*;
pub use tensor::*;
pub use unit::*;
pub use module::*;

pub mod prelude {
    pub use crate::tensor::{IntoIndexList, IntoTensor, TensorExt, TryIntoTensor};
}
