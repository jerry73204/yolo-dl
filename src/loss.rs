use crate::{
    common::*,
    model::{Detection, DetectionIndex, FeatureInfo, YoloOutput},
    utils::{BBox, RatioBBox},
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
        let (pred_cycxhw, target_cycxhw) = {
            let final_state = target_bboxes
                .iter()
                .map(|(index, target_bbox)| {
                    let prediction = &pred_detections[&index];
                    let pred_cycxhw = &prediction.cycxhw;
                    let target_cycxhw = &target_bbox.cycxhw;

                    (pred_cycxhw, target_cycxhw)
                })
                .fold((vec![], vec![]), |mut state, args| {
                    let (pred_cycxhw_vec, target_cycxhw_vec) = &mut state;
                    let (pred_cycxhw, target_cycxhw) = args;

                    pred_cycxhw_vec.push(pred_cycxhw.view([1, 4]));
                    target_cycxhw_vec.extend(target_cycxhw.iter().map(|value| value.raw() as f32));

                    state
                });

            let (pred_cycxhw_vec, target_cycxhw_vec) = final_state;
            let pred_cycxhw = Tensor::cat(&pred_cycxhw_vec, 0);
            let target_cycxhw = Tensor::of_slice(&target_cycxhw_vec)
                .view([-1, 4])
                .to_device(device);

            (pred_cycxhw, target_cycxhw)
        };

        // IoU loss
        let iou_loss: Tensor = { 1.0 - self.compute_iou(&pred_cycxhw, &target_cycxhw) };

        let reduced_iou_loss = match self.reduction {
            Reduction::Mean => iou_loss.mean(Kind::Float),
            Reduction::Sum => iou_loss.sum(Kind::Float),
            _ => panic!("reduction {:?} is not supported", self.reduction),
        };

        (reduced_iou_loss, iou_loss)
    }

    fn classification_loss(
        &self,
        pred_detections: &HashMap<GridIndex, Grid>,
        target_bboxes: &HashMap<GridIndex, Rc<BBox>>,
        num_classes: i64,
        device: Device,
    ) -> Tensor {
        // smooth bce
        let pos = 1.0 - 0.5 * self.smooth_bce_coef;
        let neg = 1.0 - pos;

        let (pred_class, target_class) = {
            let final_state = target_bboxes
                .iter()
                .map(|(index, target_bbox)| {
                    let prediction = &pred_detections[&index];
                    let pred_class = &prediction.classification;
                    let target_class = target_bbox.category_id as i64;

                    (pred_class, target_class)
                })
                .fold((vec![], vec![]), |mut state, args| {
                    let (pred_class_vec, target_class_vec) = &mut state;
                    let (pred_class, target_class) = args;

                    pred_class_vec.push(pred_class.view([-1, num_classes]));
                    target_class_vec.push(target_class);

                    state
                });

            let (pred_class_vec, target_class_vec) = final_state;
            let pred_class = Tensor::cat(&pred_class_vec, 0);
            let target_class = {
                let indexes = Tensor::of_slice(&target_class_vec).to_device(device);
                let pos_values = Tensor::full(&indexes.size(), pos, (Kind::Float, device));
                let target = pred_class
                    .full_like(neg)
                    .view([-1])
                    .scatter(0, &indexes, &pos_values)
                    .view([-1, num_classes]);
                target
            };

            (pred_class, target_class)
        };

        self.bce_class.forward(&pred_class, &target_class)
    }

    fn objectness_loss(
        &self,
        pred_detections: &HashMap<GridIndex, Grid>,
        target_bboxes: &HashMap<GridIndex, Rc<BBox>>,
        non_reduced_iou_loss: &Tensor,
        device: Device,
    ) -> Tensor {
        let target_indexes: HashSet<_> = target_bboxes.iter().map(|(index, _)| index).collect();

        let (indexes_vec, pred_vec) =
            pred_detections
                .iter()
                .enumerate()
                .fold((vec![], vec![]), |mut state, args| {
                    let (indexes_vec, pred_vec) = &mut state;
                    let (instance_index, (grid_index, detection)) = args;

                    if target_indexes.contains(grid_index) {
                        indexes_vec.push(instance_index as i64);
                    }
                    pred_vec.push(detection.objectness.view([1, 1]));

                    state
                });

        let instance_indexes = Tensor::of_slice(&indexes_vec)
            .view([-1, 1])
            .to_device(device);
        let pred_objectness = Tensor::cat(&pred_vec, 0);
        let target_objectness = pred_objectness.full_like(0.0).scatter(
            0,
            &instance_indexes,
            &(&non_reduced_iou_loss.detach().clamp(0.0, 1.0) * self.objectness_iou_ratio
                + (1.0 - self.objectness_iou_ratio)),
        );

        self.bce_objectness
            .forward(&pred_objectness, &target_objectness)
    }

    fn compute_iou(&self, pred_cycxhw: &Tensor, target_cycxhw: &Tensor) -> Tensor {
        use std::f64::consts::PI;

        let epsilon = 1e-16;

        // unpack parameters
        let pred_cy = pred_cycxhw.i((.., 0));
        let pred_cx = pred_cycxhw.i((.., 1));
        let pred_h = pred_cycxhw.i((.., 2));
        let pred_w = pred_cycxhw.i((.., 3));
        let pred_t = &pred_cy - &pred_h / 2.0;
        let pred_b = &pred_cy + &pred_h / 2.0;
        let pred_l = &pred_cx - &pred_w / 2.0;
        let pred_r = &pred_cx + &pred_w / 2.0;
        let pred_area = &pred_h * &pred_w;

        let target_cy = target_cycxhw.i((.., 0));
        let target_cx = target_cycxhw.i((.., 1));
        let target_h = target_cycxhw.i((.., 2));
        let target_w = target_cycxhw.i((.., 3));
        let target_t = &target_cy - &target_h / 2.0;
        let target_b = &target_cy + &target_h / 2.0;
        let target_l = &target_cx - &target_w / 2.0;
        let target_r = &target_cx + &target_w / 2.0;
        let target_area = &target_h * &target_w;

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

        let loss = match self.iou_kind {
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
                            (&pred_cy - &target_cy).pow(2.0) + (&pred_cx - &target_cx).pow(2.0);

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

        loss.view([-1, 1])
    }

    /// Match target bboxes with grids.
    fn build_target_bboxes(
        &self,
        prediction: &YoloOutput,
        target: &Vec<Vec<RatioBBox>>,
    ) -> HashMap<GridIndex, Rc<BBox>> {
        let feature_info = &prediction.feature_info;

        let bbox_iter = target
            .iter()
            .enumerate()
            .flat_map(|(batch_index, bboxes)| bboxes.iter().map(move |bbox| (batch_index, bbox)));

        // pair up each target bbox and each grid
        let targets: HashMap<_, _> = iproduct!(bbox_iter, feature_info.iter().enumerate())
            .flat_map(|args| {
                // unpack variables
                let ((batch_index, ratio_bbox), (layer_index, info)) = args;
                let FeatureInfo {
                    feature_height,
                    feature_width,
                    ref anchors,
                    ..
                } = *info;

                // compute bbox in grid units
                let grid_bbox = {
                    let height = R64::new(feature_height as f64);
                    let width = R64::new(feature_width as f64);
                    let bbox = ratio_bbox.to_bbox(height, width);
                    Rc::new(bbox)
                };
                let [grid_cy, grid_cx, grid_h, grid_w] = grid_bbox.cycxhw;

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

                // filter by anchor sizes
                anchors
                    .iter()
                    .cloned()
                    .enumerate()
                    .filter(move |(_index, anchor_size)| {
                        let (anchor_height, anchor_width) = *anchor_size;
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
                                let index = GridIndex {
                                    batch_index,
                                    layer_index,
                                    anchor_index,
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
    fn build_indexed_detections(prediction: &YoloOutput) -> HashMap<GridIndex, Grid> {
        prediction
            .detections
            .iter()
            .flat_map(|detection| {
                let (batch_size, _) = detection.cycxhw.size2().unwrap();

                let DetectionIndex {
                    layer_index,
                    anchor_index,
                    grid_row,
                    grid_col,
                } = detection.index;

                (0..batch_size).map(move |batch_index| {
                    let Detection {
                        cycxhw,
                        objectness,
                        classification,
                        ..
                    } = detection.shallow_clone();
                    let index = GridIndex {
                        batch_index: batch_index as usize,
                        layer_index,
                        anchor_index,
                        grid_row,
                        grid_col,
                    };
                    let grid = Grid {
                        cycxhw: cycxhw.i((batch_index, ..)),
                        objectness: objectness.i((batch_index, ..)),
                        classification: classification.i((batch_index, ..)),
                    };

                    (index, grid)
                })
            })
            .collect()
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
