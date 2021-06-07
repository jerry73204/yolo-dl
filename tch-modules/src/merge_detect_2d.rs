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
        detections: &[impl Borrow<DenseDetectionTensor>],
    ) -> Result<DenseDetectionTensorList> {
        DenseDetectionTensorList::from_detection_tensors(detections)
    }
}
