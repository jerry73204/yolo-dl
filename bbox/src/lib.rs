//! Safe bounding box types and functions.

mod common;

pub mod into_tlbr;
pub use into_tlbr::*;

pub mod into_cycxhw;
pub use into_cycxhw::*;

pub use transform::*;
mod transform;

pub use rect::*;
pub mod rect;

pub use tlbr::*;
pub mod tlbr;

pub use cycxhw::*;
pub mod cycxhw;

pub use hw::*;
pub mod hw;

pub use element::*;
pub mod element;

pub use into_hw::*;
pub mod into_hw;

#[cfg(feature = "opencv")]
mod with_opencv;

pub mod prelude {
    pub use crate::rect::{Rect, RectExt};
}
