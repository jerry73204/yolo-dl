use super::*;
use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CocoRecord {
    pub path: PathBuf,
    pub size: PixelSize<usize>,
    /// Bounding box in pixel units.
    pub bboxes: Vec<LabeledPixelBBox<R64>>,
}
