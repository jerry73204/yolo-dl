use crate::{
    common::*,
    label::{PixelLabel, RatioLabel},
};
use bbox::HW;
use tch_goodies::Pixel;

/// The record with image path and boxes, but without image pixels.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileRecord {
    pub path: PathBuf,
    pub size: Pixel<HW<usize>>,
    /// Bounding box in pixel units.
    pub bboxes: Vec<PixelLabel>,
}

/// The record with image pixels and boxes.
#[derive(Debug, TensorLike)]
pub struct DataRecord {
    pub image: Tensor,
    #[tensor_like(clone)]
    pub bboxes: Vec<RatioLabel>,
}
