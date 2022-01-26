use crate::common::*;

#[derive(Debug)]
pub struct MergeDetect2D {
    _private: [u8; 0],
}

impl MergeDetect2D {
    pub fn new() -> Self {
        Self { _private: [] }
    }

    pub fn forward(
        &mut self,
        detections: impl IntoIterator<Item = impl Borrow<DenseDetectionTensor>>,
    ) -> Result<DenseDetectionTensorList> {
        DenseDetectionTensorList::from_detection_tensors(detections)
    }
}

impl Default for MergeDetect2D {
    fn default() -> Self {
        Self::new()
    }
}
