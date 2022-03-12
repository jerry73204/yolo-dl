use super::{BoxMetric, ClassificationLossKind, MatchGrid, ObjectnessLossKind, YoloLossInit};
use crate::common::*;

/// The loss function configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// The choice of objectness loss function.
    pub objectness_loss_fn: ObjectnessLossKind,
    /// The choice of classification loss function.
    pub classification_loss_fn: ClassificationLossKind,
    /// The weight factor of positive objectness class.
    pub objectness_positive_weight: Option<R64>,
    /// The method to match ground truth to predicted bounding boxes.
    pub match_grid_method: MatchGrid,
    /// The choice of bounding box metric.
    pub box_metric: BoxMetric,
    /// The weight factor of IoU loss.
    pub iou_loss_weight: Option<R64>,
    /// The weight factor of objectness loss.
    pub objectness_loss_weight: Option<R64>,
    /// The weight factor of classification loss.
    pub classification_loss_weight: Option<R64>,
}

impl Config {
    pub fn yolo_loss_init(&self) -> YoloLossInit {
        let Self {
            box_metric,
            match_grid_method,
            iou_loss_weight,
            objectness_positive_weight,
            objectness_loss_fn,
            classification_loss_fn,
            objectness_loss_weight,
            classification_loss_weight,
        } = *self;

        let mut init = YoloLossInit {
            reduction: Reduction::Mean,
            match_grid_method,
            box_metric,
            objectness_loss_kind: objectness_loss_fn,
            classification_loss_kind: classification_loss_fn,
            objectness_pos_weight: objectness_positive_weight,
            ..Default::default()
        };

        if let Some(iou_loss_weight) = iou_loss_weight {
            init.iou_loss_weight = iou_loss_weight;
        }

        if let Some(objectness_loss_weight) = objectness_loss_weight {
            init.objectness_loss_weight = objectness_loss_weight;
        }

        if let Some(classification_loss_weight) = classification_loss_weight {
            init.classification_loss_weight = classification_loss_weight;
        }

        init
    }
}
