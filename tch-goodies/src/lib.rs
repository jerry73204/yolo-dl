//! The crate provides extension to [Tensor](tch::Tensor) type and utilities.

mod common;
pub mod compound_tensor;
pub mod detection;
pub mod lr_schedule;
pub mod tensor;
pub mod tensor_list;
pub mod unit;
mod utils;

pub use compound_tensor::*;
pub use detection::*;
pub use tensor::*;
pub use tensor_list::*;
pub use unit::*;

pub mod prelude {
    pub use crate::tensor::{IntoIndexList, TensorExt};
}
