//! The crate provides extension to [Tensor](tch::Tensor) type and utilities.

pub mod bbox;
mod common;
pub mod detection;
pub mod ratio;
pub mod size;
pub mod tensor;
pub mod unit;

pub use bbox::*;
pub use detection::*;
pub use ratio::*;
pub use size::*;
pub use tensor::*;
pub use unit::*;

pub mod prelude {
    pub use crate::tensor::{IntoIndexList, IntoTensor, TensorExt, TryIntoTensor};
}
