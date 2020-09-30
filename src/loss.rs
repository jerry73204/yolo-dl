use crate::{
    common::*,
    model::{InstanceIndex, LayerMeta, YoloOutput},
    utils::{GridSize, LabeledGridBBox, LabeledRatioBBox, Unzip5},
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MatchGrid {
    Rect2,
    Rect4,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
    pub fn forward(&self, prediction: &YoloOutput, target: &Vec<Vec<LabeledRatioBBox>>) -> Tensor {
        // match target bboxes and grids, and group them by detector cells
        // indexed by grid positions
        let target_bboxes = self.match_target_bboxes(&prediction, target);

        // collect selected instances
        let (pred_instances, target_instances) =
            Self::collect_instances(prediction, &target_bboxes);

        // IoU loss
        let (iou_loss, non_reduced_iou_loss) = self.iou_loss(&pred_instances, &target_instances);

        // classification loss
        let classification_loss = self.classification_loss(&pred_instances, &target_instances);

        // objectness loss
        let objectness_loss =
            self.objectness_loss(&prediction, &target_bboxes, &non_reduced_iou_loss);

        // normalize and balancing
        let loss = self.iou_loss_weight * &iou_loss
            + self.classification_loss_weight * &classification_loss
            + self.objectness_loss_weight * &objectness_loss;

        // let loss = self.iou_loss_weight * &iou_loss;
        // let loss = self.classification_loss_weight * &classification_loss;
        // let loss = self.objectness_loss_weight * &objectness_loss;

        loss
    }

    fn iou_loss(
        &self,
        pred_instances: &PredInstances,
        target_instances: &TargetInstances,
    ) -> (Tensor, Tensor) {
        use std::f64::consts::PI;
        let epsilon = 1e-16;

        // prediction bbox properties
        let PredInstances {
            cy: pred_cy,
            cx: pred_cx,
            height: pred_h,
            width: pred_w,
            ..
        } = pred_instances;

        let pred_t = pred_cy - pred_h / 2.0;
        let pred_b = pred_cy + pred_h / 2.0;
        let pred_l = pred_cx - pred_w / 2.0;
        let pred_r = pred_cx + pred_w / 2.0;
        let pred_area = pred_h * pred_w;

        // target bbox properties
        let TargetInstances {
            cy: target_cy,
            cx: target_cx,
            height: target_h,
            width: target_w,
            ..
        } = target_instances;

        let target_t = target_cy - target_h / 2.0;
        let target_b = target_cy + target_h / 2.0;
        let target_l = target_cx - target_w / 2.0;
        let target_r = target_cx + target_w / 2.0;
        let target_area = target_h * target_w;

        // compute intersection area
        let intersect_t = pred_t.max1(&target_t);
        let intersect_l = pred_l.max1(&target_l);
        let intersect_b = pred_b.min1(&target_b);
        let intersect_r = pred_r.min1(&target_r);
        let intersect_h = &intersect_b - &intersect_t;
        let intersect_w = &intersect_r - &intersect_l;
        let intersect_area = &intersect_h * &intersect_w;

        // compute IoU
        let union_area = &pred_area + &target_area - &intersect_area + epsilon;
        let iou = &intersect_area / &union_area; // TODO: plus epsilon

        let xiou = match self.iou_kind {
            IoUKind::IoU => iou,
            _ => {
                let closure_t = pred_t.min1(&target_t);
                let closure_l = pred_l.min1(&target_l);
                let closure_b = pred_b.max1(&target_b);
                let closure_r = pred_r.max1(&target_r);
                let closure_h = &closure_b - &closure_t;
                let closure_w = &closure_r - &closure_l;
                let closure_area = &closure_h * &closure_w + epsilon;

                match self.iou_kind {
                    IoUKind::GIoU => &iou - (&closure_area - &union_area) / &closure_area,
                    _ => {
                        let diagonal_square = closure_h.pow(2.0) + closure_w.pow(2.0) + epsilon;
                        let center_dist_square =
                            (pred_cy - target_cy).pow(2.0) + (pred_cx - target_cx).pow(2.0);

                        match self.iou_kind {
                            IoUKind::DIoU => &iou - &center_dist_square / &diagonal_square,
                            IoUKind::CIoU => {
                                let pred_angle = pred_h.atan2(&pred_w);
                                let target_angle = target_h.atan2(&target_w);

                                let shape_loss =
                                    (&pred_angle - &target_angle).pow(2.0) * 4.0 / PI.powi(2);
                                let shape_loss_coef =
                                    tch::no_grad(|| &shape_loss / (1.0 - &iou + &shape_loss));

                                &iou - &center_dist_square / &diagonal_square
                                    + &shape_loss_coef * &shape_loss
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }
        };

        // IoU loss
        let iou_loss: Tensor = { 1.0 - xiou };
        let reduced_iou_loss = match self.reduction {
            Reduction::Mean => iou_loss.mean(Kind::Float),
            Reduction::Sum => iou_loss.sum(Kind::Float),
            _ => panic!("reduction {:?} is not supported", self.reduction),
        };

        (reduced_iou_loss, iou_loss)
    }

    fn classification_loss(
        &self,
        pred_instances: &PredInstances,
        target_instances: &TargetInstances,
    ) -> Tensor {
        // smooth bce values
        let pos = 1.0 - 0.5 * self.smooth_bce_coef;
        let neg = 1.0 - pos;
        let pred_class = &pred_instances.dense_class;
        let (_num_instances, num_classes) = pred_class.size2().unwrap();
        let device = pred_class.device();

        // convert sparse index to dense one-hot-like
        let target_class_dense = {
            let target_class_sparse = &target_instances.sparse_class;
            let pos_values = Tensor::full(&target_class_sparse.size(), pos, (Kind::Float, device));
            let target = pred_class
                .full_like(neg)
                .scatter(0, &target_class_sparse, &pos_values)
                .view([-1, num_classes]);
            target
        };

        self.bce_class.forward(&pred_class, &target_class_dense)
    }

    fn objectness_loss(
        &self,
        prediction: &YoloOutput,
        target_bboxes: &HashMap<InstanceIndex, Rc<LabeledGridBBox<R64>>>,
        non_reduced_iou_loss: &Tensor,
    ) -> Tensor {
        let device = prediction.device;
        let pred_flat_indexes = Tensor::of_slice(
            &target_bboxes
                .keys()
                .map(|instance_index| prediction.to_flat_index(instance_index))
                .collect_vec(),
        )
        .view([-1, 1])
        .to_device(device);

        let pred_objectness = &prediction.objectness;
        let target_objectness = {
            let mut target = pred_objectness.full_like(0.0);
            let _ = target.scatter_(
                0,
                &pred_flat_indexes,
                &(&non_reduced_iou_loss.detach().clamp(0.0, 1.0) * self.objectness_iou_ratio
                    + (1.0 - self.objectness_iou_ratio)),
            );
            target
        };

        self.bce_objectness
            .forward(&pred_objectness, &target_objectness)
    }

    /// Match target bboxes with grids.
    fn match_target_bboxes(
        &self,
        prediction: &YoloOutput,
        target: &Vec<Vec<LabeledRatioBBox>>,
    ) -> HashMap<InstanceIndex, Rc<LabeledGridBBox<R64>>> {
        let bbox_iter = target
            .iter()
            .enumerate()
            .flat_map(|(batch_index, bboxes)| bboxes.iter().map(move |bbox| (batch_index, bbox)));

        // pair up each target bbox and each grid
        let targets: HashMap<_, _> = iproduct!(bbox_iter, prediction.layer_meta.iter().enumerate())
            .flat_map(|args| {
                // unpack variables
                let ((batch_index, ratio_bbox), (layer_index, layer)) = args;
                let LayerMeta {
                    feature_size:
                        GridSize {
                            height: feature_height,
                            width: feature_width,
                            ..
                        },
                    ref anchors,
                    ..
                } = *layer;

                // compute bbox in grid units
                let grid_bbox = {
                    let height = R64::new(feature_height as f64);
                    let width = R64::new(feature_width as f64);
                    Rc::new(LabeledGridBBox {
                        bbox: ratio_bbox.bbox.to_bbox(height, width),
                        category_id: ratio_bbox.category_id,
                    })
                };
                let [grid_cy, grid_cx, grid_h, grid_w] = grid_bbox.bbox.cycxhw;

                // collect neighbor grid indexes
                let grid_indexes = {
                    let margin_thresh = 0.5;
                    let grid_row = grid_cy.floor().raw() as i64;
                    let grid_col = grid_cx.floor().raw() as i64;
                    debug_assert!(grid_row >= 0 && grid_col >= 0);

                    let grid_indexes: Vec<_> = {
                        let orig_iter = iter::once((grid_row, grid_col));
                        match self.match_grid_method {
                            MatchGrid::Rect2 => {
                                let top_iter = if grid_cy % 1.0 < margin_thresh && grid_row > 0 {
                                    Some((grid_row - 1, grid_col))
                                } else {
                                    None
                                }
                                .into_iter();
                                let left_iter = if grid_cx < margin_thresh && grid_col > 0 {
                                    Some((grid_row, grid_col - 1))
                                } else {
                                    None
                                }
                                .into_iter();

                                orig_iter.chain(top_iter).chain(left_iter).collect()
                            }
                            MatchGrid::Rect4 => {
                                let top_iter = if grid_cy % 1.0 < margin_thresh && grid_row > 0 {
                                    Some((grid_row - 1, grid_col))
                                } else {
                                    None
                                }
                                .into_iter();
                                let left_iter = if grid_cx < margin_thresh && grid_col > 0 {
                                    Some((grid_row, grid_col - 1))
                                } else {
                                    None
                                }
                                .into_iter();
                                let bottom_iter = if grid_cy % 1.0 > (1.0 - margin_thresh)
                                    && grid_row <= feature_height - 2
                                {
                                    Some((grid_row + 1, grid_col))
                                } else {
                                    None
                                }
                                .into_iter();
                                let right_iter = if grid_cx % 1.0 < (1.0 - margin_thresh)
                                    && grid_col <= feature_width - 2
                                {
                                    Some((grid_row, grid_col + 1))
                                } else {
                                    None
                                }
                                .into_iter();

                                orig_iter
                                    .chain(top_iter)
                                    .chain(left_iter)
                                    .chain(bottom_iter)
                                    .chain(right_iter)
                                    .collect()
                            }
                        }
                    };

                    grid_indexes
                };

                anchors
                    .iter()
                    .cloned()
                    .enumerate()
                    .filter(move |(_index, anchor_size)| {
                        // filter by anchor sizes
                        let GridSize {
                            height: anchor_height,
                            width: anchor_width,
                            ..
                        } = *anchor_size;
                        let grid_h = grid_h.raw();
                        let grid_w = grid_w.raw();

                        grid_h / anchor_height <= self.anchor_scale_thresh
                            && anchor_height / grid_h <= self.anchor_scale_thresh
                            && grid_w / anchor_width <= self.anchor_scale_thresh
                            && anchor_width / grid_w <= self.anchor_scale_thresh
                    })
                    .flat_map(move |(anchor_index, _)| {
                        let grid_bbox = grid_bbox.clone();
                        let targets: Vec<_> = grid_indexes
                            .iter()
                            .cloned()
                            .map(move |(grid_row, grid_col)| {
                                let index = InstanceIndex {
                                    batch_index,
                                    layer_index,
                                    anchor_index: anchor_index as i64,
                                    grid_row,
                                    grid_col,
                                };

                                (index, grid_bbox.clone())
                            })
                            .collect();

                        targets
                    })
            })
            .collect();

        targets
    }

    /// Build HashSet of predictions indexed by per-grid position
    fn collect_instances(
        prediction: &YoloOutput,
        target: &HashMap<InstanceIndex, Rc<LabeledGridBBox<R64>>>,
    ) -> (PredInstances, TargetInstances) {
        let device = prediction.device;
        let pred_instances = {
            let flat_indexes = Tensor::of_slice(
                &target
                    .keys()
                    .map(|instance_index| prediction.to_flat_index(instance_index))
                    .collect_vec(),
            )
            .to_device(device);

            let cy = prediction.cy.index_select(0, &flat_indexes);
            let cx = prediction.cx.index_select(0, &flat_indexes);
            let height = prediction.height.index_select(0, &flat_indexes);
            let width = prediction.width.index_select(0, &flat_indexes);
            let objectness = prediction.objectness.index_select(0, &flat_indexes);
            let classification = prediction.classification.index_select(0, &flat_indexes);

            PredInstances {
                cy,
                cx,
                height,
                width,
                objectness,
                dense_class: classification,
            }
        };

        let target_instances = {
            let (cy_vec, cx_vec, h_vec, w_vec, category_id_vec) = target
                .values()
                .map(|bbox| {
                    let [cy, cx, h, w] = bbox.bbox.cycxhw;
                    let category_id = bbox.category_id;
                    bbox.category_id;
                    (
                        cy.raw() as f32,
                        cx.raw() as f32,
                        h.raw() as f32,
                        w.raw() as f32,
                        category_id as i64,
                    )
                })
                .unzip_n_vec();

            let cy = Tensor::of_slice(&cy_vec).view([-1, 1]).to_device(device);
            let cx = Tensor::of_slice(&cx_vec).view([-1, 1]).to_device(device);
            let height = Tensor::of_slice(&h_vec).view([-1, 1]).to_device(device);
            let width = Tensor::of_slice(&w_vec).view([-1, 1]).to_device(device);
            let category_id = Tensor::of_slice(&category_id_vec)
                .view([-1, 1])
                .to_device(device);

            TargetInstances {
                cy,
                cx,
                height,
                width,
                sparse_class: category_id,
            }
        };

        (pred_instances, target_instances)
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

#[derive(Debug, TensorLike)]
struct Grid {
    pub cycxhw: Tensor,
    pub objectness: Tensor,
    pub classification: Tensor,
}

#[derive(Debug, TensorLike)]
pub struct PredInstances {
    pub cy: Tensor,
    pub cx: Tensor,
    pub height: Tensor,
    pub width: Tensor,
    pub objectness: Tensor,
    pub dense_class: Tensor,
}

#[derive(Debug, TensorLike)]
pub struct TargetInstances {
    pub cy: Tensor,
    pub cx: Tensor,
    pub height: Tensor,
    pub width: Tensor,
    pub sparse_class: Tensor,
}
