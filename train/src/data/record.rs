use crate::common::*;

/// The record with image path and boxes, but without image pixels.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FileRecord {
    pub path: PathBuf,
    pub size: PixelSize<usize>,
    /// Bounding box in pixel units.
    pub bboxes: Vec<LabeledPixelCyCxHW<R64>>,
}

/// The record with image pixels and boxes.
#[derive(Debug, TensorLike)]
pub struct DataRecord {
    pub image: Tensor,
    #[tensor_like(clone)]
    pub bboxes: Vec<LabeledRatioCyCxHW>,
}

/// The record that is accepted by training worker.
#[derive(Debug, TensorLike)]
pub struct TrainingRecord {
    pub epoch: usize,
    pub step: usize,
    pub image: Tensor,
    #[tensor_like(clone)]
    pub bboxes: Vec<Vec<LabeledRatioCyCxHW>>,
    #[tensor_like(clone)]
    pub timing: Timing,
}
