use super::*;
use crate::{common::*, detection::MergedDenseDetection};

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
        detections: &[impl Borrow<Detect2DOutput>],
    ) -> Result<MergeDetect2DOutput> {
        MergedDenseDetection::from_detection_tensors(detections)
    }
}

pub type MergeDetect2DOutput = MergedDenseDetection;
