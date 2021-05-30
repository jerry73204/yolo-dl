use crate::{
    common::*,
    detection::{DenseDetectionTensor, DenseDetectionTensorList},
};

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
    ) -> DenseDetectionTensorList {
        let tensors: Vec<_> = detections
            .iter()
            .map(|det| det.borrow().shallow_clone())
            .collect();
        DenseDetectionTensorList { tensors }
    }
}
