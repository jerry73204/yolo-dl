use super::nms::{NmsOutput, NonMaxSuppression, NonMaxSuppressionInit};
use crate::{common::*, model::MergeDetect2DOutput};

#[derive(Debug)]
pub struct YoloInferenceInit {
    pub nms_iou_thresh: R64,
    pub nms_conf_thresh: R64,
}

impl YoloInferenceInit {
    pub fn build(self) -> Result<YoloInference> {
        let Self {
            nms_iou_thresh,
            nms_conf_thresh,
        } = self;

        let nms = NonMaxSuppressionInit {
            iou_threshold: nms_iou_thresh,
            confidence_threshold: nms_conf_thresh,
        }
        .build()?;

        Ok(YoloInference { nms })
    }
}

#[derive(Debug)]
pub struct YoloInference {
    nms: NonMaxSuppression,
}

impl YoloInference {
    pub fn forward(&self, prediction: &MergeDetect2DOutput) -> YoloInferenceOutput {
        tch::no_grad(|| {
            let device = prediction.device();

            // run nms
            let NmsOutput {
                batches,
                classes,
                instances,
                bbox,
                confidence,
            } = self.nms.forward(prediction);

            let selected = {
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
            };

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
        })
    }
}

#[derive(Debug, TensorLike)]
pub struct YoloInferenceOutput {
    /// Batch indexes in shape `[object]`.
    pub batches: Tensor,
    /// Class indexes in shape `[object]`.
    pub classes: Tensor,
    /// Object indexes in shape `[object]`.
    pub instances: Tensor,
    /// Box parameters.
    pub bbox: TLBRTensor,
    /// Confidence scores in shape `[object, 1]`.
    pub confidence: Tensor,
}

impl YoloInferenceOutput {
    pub fn num_samples(&self) -> i64 {
        self.batches.size1().unwrap()
    }

    pub fn device(&self) -> Device {
        self.batches.device()
    }
}
