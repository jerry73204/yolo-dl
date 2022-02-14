//! Safe bounding box types and functions.

const EPSILON: f64 = 1e-16;

pub use transform::*;
mod transform;

pub use rect::*;
mod rect;

pub use tlbr::*;
mod tlbr;

pub use cycxhw::*;
mod cycxhw;

pub use rect_label::*;
mod rect_label;

#[cfg(feature = "opencv")]
mod with_opencv;
