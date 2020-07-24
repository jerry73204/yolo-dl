use crate::{
    common::*,
    model::{Detection, DetectionIndex, FeatureInfo, YoloOutput},
    utils::{BBox, RatioBBox, Unzip3},
};

#[derive(Debug)]
pub struct YoloLossInit {
    pub pos_weight: Option<Tensor>,
    pub reduction: Reduction,
    pub focal_loss_gamma: Option<f64>,
    pub match_grid_method: Option<MatchGrid>,
    pub iou_kind: Option<IoUKind>,
    pub smooth_bce_coef: Option<f64>,
    pub objectness_iou_ratio: Option<f64>,
    pub anchor_scale_thresh: Option<f64>,
    pub iou_loss_weight: Option<f64>,
    pub objectness_loss_weight: Option<f64>,
    pub classification_loss_weight: Option<f64>,
}

impl YoloLossInit {
    pub fn build(self) -> YoloLoss {
        let Self {
            pos_weight,
            reduction,
            focal_loss_gamma,
            match_grid_method,
            iou_kind,
            smooth_bce_coef,
            objectness_iou_ratio,
            anchor_scale_thresh,
            iou_loss_weight,
            objectness_loss_weight,
            classification_loss_weight,
        } = self;

        let match_grid_method = match_grid_method.unwrap_or(MatchGrid::Rect4);
        let focal_loss_gamma = focal_loss_gamma.unwrap_or(0.0);
        let iou_kind = iou_kind.unwrap_or(IoUKind::GIoU);
        let smooth_bce_coef = smooth_bce_coef.unwrap_or(0.01);
        let objectness_iou_ratio = objectness_iou_ratio.unwrap_or(1.0);
        let anchor_scale_thresh = anchor_scale_thresh.unwrap_or(4.0);
        let iou_loss_weight = iou_loss_weight.unwrap_or(0.05);
        let objectness_loss_weight = objectness_loss_weight.unwrap_or(1.0);
        let classification_loss_weight = classification_loss_weight.unwrap_or(0.58);

        assert!(focal_loss_gamma >= 0.0);
        assert!(smooth_bce_coef >= 0.0 && smooth_bce_coef <= 1.0);
        assert!(objectness_iou_ratio >= 0.0 && objectness_iou_ratio <= 1.0);
        assert!(anchor_scale_thresh >= 1.0);
        assert!(iou_loss_weight >= 0.0);
        assert!(objectness_loss_weight >= 0.0);
        assert!(classification_loss_weight >= 0.0);

        let bce_class = FocalLossInit {
            pos_weight: pos_weight.as_ref().map(|weight| weight.shallow_clone()),
            gamma: focal_loss_gamma,
            reduction: Reduction::Sum,
            ..Default::default()
        }
        .build();

        let bce_objectness = FocalLossInit {
            pos_weight,
            gamma: focal_loss_gamma,
            reduction,
            ..Default::default()
        }
        .build();

        YoloLoss {
            reduction,
            bce_class,
            bce_objectness,
            match_grid_method,
            iou_kind,
            smooth_bce_coef,
            objectness_iou_ratio,
            anchor_scale_thresh,
            iou_loss_weight,
            objectness_loss_weight,
            classification_loss_weight,
        }
    }
}

impl Default for YoloLossInit {
    fn default() -> Self {
        Self {
            pos_weight: None,
            reduction: Reduction::Mean,
            focal_loss_gamma: None,
            match_grid_method: None,
            iou_kind: None,
            smooth_bce_coef: None,
            objectness_iou_ratio: None,
            anchor_scale_thresh: None,
            iou_loss_weight: None,
            objectness_loss_weight: None,
            classification_loss_weight: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MatchGrid {
    Rect2,
    Rect4,
}

#[derive(Debug, Clone, Copy)]
pub enum IoUKind {
    IoU,
    GIoU,
    DIoU,
    CIoU,
}

#[derive(Debug)]
pub struct YoloLoss {
    reduction: Reduction,
    bce_class: FocalLoss,
    bce_objectness: FocalLoss,
    match_grid_method: MatchGrid,
    iou_kind: IoUKind,
    smooth_bce_coef: f64,
    objectness_iou_ratio: f64,
    anchor_scale_thresh: f64,
    iou_loss_weight: f64,
    objectness_loss_weight: f64,
    classification_loss_weight: f64,
}

impl YoloLoss {
    pub fn forward(&self, prediction: &YoloOutput, target: &Vec<Vec<RatioBBox>>) -> Tensor {
        let device = prediction.device;
        let num_classes = prediction.num_classes;

        // match target bboxes and grids, and group them by detector cells
        // indexed by grid positions
        let target_bboxes: HashMap<GridIndex, Rc<BBox>> =
            self.build_target_bboxes(prediction, target);

        // index detections
        let pred_detections = Self::build_indexed_detections(prediction);

        // IoU loss
        let (iou_loss, non_reduced_iou_loss) =
            self.iou_loss(&pred_detections, &target_bboxes, device);

        // classification loss
        let classification_loss =
            self.classification_loss(&pred_detections, &target_bboxes, num_classes, device);

        // objectness loss
        let objectness_loss = self.objectness_loss(
            &pred_detections,
            &target_bboxes,
            &non_reduced_iou_loss,
            device,
        );

        // normalize and balancing
        let loss = self.iou_loss_weight * &iou_loss
            + self.classification_loss_weight * &classification_loss
            + self.objectness_loss_weight * &objectness_loss;

        loss
    }

    fn iou_loss(
        &self,
        pred_detections: &HashMap<GridIndex, Grid>,
        target_bboxes: &HashMap<GridIndex, Rc<BBox>>,
        device: Device,
    ) -> (Tensor, Tensor) {
        todo!();
    }

    fn classification_loss(
        &self,
        pred_detections: &HashMap<GridIndex, Grid>,
        target_bboxes: &HashMap<GridIndex, Rc<BBox>>,
        num_classes: i64,
        device: Device,
    ) -> Tensor {
        todo!();
    }

    fn objectness_loss(
        &self,
        pred_detections: &HashMap<GridIndex, Grid>,
        target_bboxes: &HashMap<GridIndex, Rc<BBox>>,
        non_reduced_iou_loss: &Tensor,
        device: Device,
    ) -> Tensor {
        todo!();
    }

    fn compute_iou(&self, pred_cycxhw: &Tensor, target_cycxhw: &Tensor) -> Tensor {
        todo!();
    }

    /// Match target bboxes with grids.
    fn build_target_bboxes(
        &self,
        prediction: &YoloOutput,
        target: &Vec<Vec<RatioBBox>>,
    ) -> HashMap<GridIndex, Rc<BBox>> {
        // convert target to tensors
        let (batch_index_vec, cycxhw_components_vec, category_id_vec) = target
            .iter()
            .enumerate()
            .flat_map(|(batch_index, bboxes)| {
                bboxes.iter().map(move |bbox| {
                    let cycxhw_components: Vec<_> =
                        bbox.cycxhw.iter().cloned().map(|comp| comp.raw()).collect();

                    (
                        batch_index as i64,
                        cycxhw_components,
                        bbox.category_id as i64,
                    )
                })
            })
            .unzip_n_vec();

        let target_batch_indexes = Tensor::of_slice(&batch_index_vec).view([-1, 1]);
        let target_ratio_cycxhw = {
            let components: Vec<_> = cycxhw_components_vec
                .into_iter()
                .flat_map(|vec| vec)
                .collect();
            Tensor::of_slice(&components).view([-1, 4])
        };
        let target_classes = Tensor::of_slice(&category_id_vec).view([-1, 1]);

        //

        todo!();
    }

    /// Build HashSet of predictions indexed by per-grid position
    fn build_indexed_detections(prediction: &YoloOutput) -> HashMap<GridIndex, Grid> {
        todo!();
    }
}

#[derive(Debug)]
pub struct BceWithLogitsLossInit {
    pub weight: Option<Tensor>,
    pub pos_weight: Option<Tensor>,
    pub reduction: Reduction,
}

impl BceWithLogitsLossInit {
    pub fn build(self) -> BceWithLogitsLoss {
        let Self {
            weight,
            pos_weight,
            reduction,
        } = self;

        BceWithLogitsLoss {
            weight,
            pos_weight,
            reduction,
        }
    }
}

impl Default for BceWithLogitsLossInit {
    fn default() -> Self {
        Self {
            weight: None,
            pos_weight: None,
            reduction: Reduction::Mean,
        }
    }
}

#[derive(Debug)]
pub struct BceWithLogitsLoss {
    weight: Option<Tensor>,
    pos_weight: Option<Tensor>,
    reduction: Reduction,
}

impl BceWithLogitsLoss {
    pub fn forward(&self, input: &Tensor, target: &Tensor) -> Tensor {
        input.binary_cross_entropy_with_logits(
            target,
            self.weight.as_ref(),
            self.pos_weight.as_ref(),
            self.reduction,
        )
    }
}

#[derive(Debug)]
pub struct FocalLossInit {
    pub weight: Option<Tensor>,
    pub pos_weight: Option<Tensor>,
    pub gamma: f64,
    pub alpha: f64,
    pub reduction: Reduction,
}

impl FocalLossInit {
    pub fn build(self) -> FocalLoss {
        let Self {
            weight,
            pos_weight,
            gamma,
            alpha,
            reduction,
        } = self;

        let bce = BceWithLogitsLossInit {
            weight,
            pos_weight,
            reduction: Reduction::None,
        }
        .build();

        FocalLoss {
            bce,
            gamma,
            alpha,
            reduction,
        }
    }
}

impl Default for FocalLossInit {
    fn default() -> Self {
        Self {
            weight: None,
            pos_weight: None,
            gamma: 1.5,
            alpha: 0.25,
            reduction: Reduction::Mean,
        }
    }
}

#[derive(Debug)]
pub struct FocalLoss {
    bce: BceWithLogitsLoss,
    gamma: f64,
    alpha: f64,
    reduction: Reduction,
}

impl FocalLoss {
    pub fn forward(&self, input: &Tensor, target: &Tensor) -> Tensor {
        let Self {
            ref bce,
            gamma,
            alpha,
            reduction,
        } = *self;

        let bce_loss = bce.forward(input, target);
        let input_prob = input.sigmoid();
        let p_t: Tensor = target * &input_prob + (1.0 - target) * (1.0 - &input_prob);
        let alpha_factor = target * alpha + (1.0 - target) * (1.0 - alpha);
        let modulating_factor = (-&p_t + 1.0).pow(gamma);
        let loss: Tensor = bce_loss * alpha_factor * modulating_factor;

        match reduction {
            Reduction::Mean => loss.mean(Kind::Float),
            Reduction::Sum => loss.sum(Kind::Float),
            Reduction::None => loss,
            Reduction::Other(_) => unreachable!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, TensorLike)]
struct GridIndex {
    pub batch_index: usize,
    pub layer_index: usize,
    pub anchor_index: usize,
    pub grid_row: i64,
    pub grid_col: i64,
}

#[derive(Debug, TensorLike)]
struct Grid {
    pub cycxhw: Tensor,
    pub objectness: Tensor,
    pub classification: Tensor,
}
