use super::*;
use crate::common::*;

// record types

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileRecord {
    pub path: PathBuf,
    pub size: PixelSize<usize>,
    /// Bounding box in pixel units.
    pub bboxes: Vec<LabeledPixelBBox<R64>>,
}

#[derive(Debug, TensorLike)]
pub struct DataRecord {
    pub image: Tensor,
    #[tensor_like(clone)]
    pub bboxes: Vec<LabeledRatioBBox>,
}

#[derive(Debug, TensorLike)]
pub struct TrainingRecord {
    pub epoch: usize,
    pub step: usize,
    pub image: Tensor,
    #[tensor_like(clone)]
    pub bboxes: Vec<Vec<LabeledRatioBBox>>,
}
