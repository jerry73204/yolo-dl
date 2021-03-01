use super::nms::{NmsOutput, NonMaxSuppression, NonMaxSuppressionInit};
use crate::{common::*, model::MergeDetect2DOutput};

#[derive(Debug)]
pub struct YoloInferenceInit {
    pub nms_iou_threshold: Option<R64>,
    pub nms_confidence_threshold: Option<R64>,
}

impl YoloInferenceInit {
    pub fn build(self) -> Result<YoloInference> {
        let Self {
            nms_iou_threshold,
            nms_confidence_threshold,
        } = self;

        let nms = {
            let mut init = NonMaxSuppressionInit::default();
            if let Some(iou_threshold) = nms_iou_threshold {
                init.iou_threshold = iou_threshold;
            }
            if let Some(confidence_threshold) = nms_confidence_threshold {
                init.confidence_threshold = confidence_threshold;
            }
            init.build()?
        };

        Ok(YoloInference { nms })
    }
}

impl Default for YoloInferenceInit {
    fn default() -> Self {
        Self {
            nms_iou_threshold: None,
            nms_confidence_threshold: None,
        }
    }
}

#[derive(Debug)]
pub struct YoloInference {
    nms: NonMaxSuppression,
}

impl YoloInference {
    pub fn forward(&self, prediction: &MergeDetect2DOutput) -> YoloInferenceOutput {
        // run nms
        let NmsOutput {
            batches,
            classes,
            instances,
            bbox,
        } = self.nms.forward(prediction);

        YoloInferenceOutput {
            batches,
            classes,
            instances,
            bbox,
        }
    }
}

#[derive(Debug, TensorLike)]
pub struct YoloInferenceOutput {
    pub batches: Tensor,
    pub classes: Tensor,
    pub instances: Tensor,
    pub bbox: TLBRTensor,
}
