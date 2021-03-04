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
        let device = prediction.device();

        // run nms
        let NmsOutput {
            batches,
            classes,
            instances,
            bbox,
            confidence,
        } = self.nms.forward(prediction);

        let selected = tch::no_grad(|| {
            let selected: Vec<i64> = izip!(
                Vec::<i64>::from(&batches),
                Vec::<i64>::from(&classes),
                Vec::<i64>::from(&instances),
                Vec::<f32>::from(&confidence)
            )
            .enumerate()
            // group up samples by (batch_index, instance_index)
            .map(|args| {
                let (nms_index, (batch, class, instance, confidence)) = args;
                ((batch, instance), (nms_index, class, r32(confidence)))
            })
            .into_group_map()
            .into_iter()
            // for each samples of the same (batch_index, instance_index),
            // pick the one with max confidence.
            .map(|args| {
                let ((_batch, _instance), triples) = args;
                let (nms_index, _class, _confidence) = triples
                    .into_iter()
                    .max_by_key(|(_nms_index, _class, confidence)| *confidence)
                    .unwrap();
                nms_index as i64
            })
            .collect();
            Tensor::of_slice(&selected).to_device(device)
        });

        let selected_batches = batches.index(&[&selected]);
        let selected_classes = classes.index(&[&selected]);
        let selected_instances = instances.index(&[&selected]);
        let selected_bbox = bbox.index_select(&selected);
        let selected_confidence = confidence.index(&[&selected]);

        YoloInferenceOutput {
            batches: selected_batches,
            classes: selected_classes,
            instances: selected_instances,
            bbox: selected_bbox,
            confidence: selected_confidence,
        }
    }
}

#[derive(Debug, TensorLike)]
pub struct YoloInferenceOutput {
    pub batches: Tensor,
    pub classes: Tensor,
    pub instances: Tensor,
    pub bbox: TLBRTensor,
    pub confidence: Tensor,
}

impl YoloInferenceOutput {
    pub fn device(&self) -> Device {
        self.batches.device()
    }
}
