//! Unit marker types.

use crate::{common::*, ratio::Ratio};

pub trait Unit {
    type Type;
}
impl Unit for PixelUnit {
    type Type = usize;
}
impl Unit for GridUnit {
    type Type = R64;
}
impl Unit for RatioUnit {
    type Type = Ratio;
}

/// The pixel unit marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TensorLike)]
pub struct PixelUnit;

/// The ratio unit marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TensorLike)]
pub struct RatioUnit;

/// The grid unit marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TensorLike)]
pub struct GridUnit;
