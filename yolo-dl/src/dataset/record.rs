use crate::common::*;
use tch_goodies::{PixelRectLabel, PixelSize, RatioRectLabel};

/// The record with image path and boxes, but without image pixels.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileRecord {
    pub path: PathBuf,
    pub size: PixelSize<usize>,
    /// Bounding box in pixel units.
    pub bboxes: Vec<PixelRectLabel<R64>>,
}

/// The record with image pixels and boxes.
#[derive(Debug, TensorLike)]
pub struct DataRecord {
    pub image: Tensor,
    #[tensor_like(clone)]
    pub bboxes: Vec<RatioRectLabel<R64>>,
}
