//! Unit marker types.

use crate::common::*;

pub trait Unit {}
impl Unit for PixelUnit {}
impl Unit for GridUnit {}
impl Unit for RatioUnit {}

/// The pixel unit marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TensorLike)]
pub struct PixelUnit;

/// The ratio unit marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TensorLike)]
pub struct RatioUnit;

/// The grid unit marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TensorLike)]
pub struct GridUnit;
