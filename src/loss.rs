use crate::{
    common::*,
    model::{Detection, DetectionIndex, YoloOutput},
    utils::{BBox, RatioBBox},
};

#[derive(Debug)]
pub struct YoloLossInit {
    pub pos_weight: Option<Tensor>,
    pub reduction: Reduction,
    pub focal_loss_gamma: Option<f64>,
    pub anchors_list: Vec<Vec<(usize, usize)>>,
    pub match_grid_method: Option<MatchGrid>,
    pub iou_kind: Option<IoUKind>,
}

impl YoloLossInit {
    pub fn build(self) -> YoloLoss {
        let Self {
            pos_weight,
            reduction,
            focal_loss_gamma,
            anchors_list,
            match_grid_method,
            iou_kind,
        } = self;
        let match_grid_method = match_grid_method.unwrap_or(MatchGrid::Rect4);
        let focal_loss_gamma = focal_loss_gamma.unwrap_or(0.0);
        let iou_kind = iou_kind.unwrap_or(IoUKind::GIoU);
        assert!(focal_loss_gamma >= 0.0);

        let bce_class = FocalLossInit {
            pos_weight: pos_weight.as_ref().map(|weight| weight.shallow_clone()),
            gamma: focal_loss_gamma,
            reduction,
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

        let gain = Tensor::ones(&[6], (Kind::Float, Device::Cpu));
        let offsets = Tensor::of_slice(&[1, 0, 0, 1, -1, 0, 0, -1]).view([3, 2]);
        let anchor_tensors = Tensor::arange(anchors_list.len() as i64, (Kind::Int64, Device::Cpu));

        YoloLoss {
            bce_class,
            bce_objectness,
            anchors_list,
            match_grid_method,
            iou_kind,
        }
    }
}

impl YoloLossInit {
    pub fn new(anchors_list: Vec<Vec<(usize, usize)>>) -> Self {
        Self {
            pos_weight: None,
            reduction: Reduction::Mean,
            focal_loss_gamma: None,
            anchors_list,
            match_grid_method: None,
            iou_kind: None,
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
    bce_class: FocalLoss,
    bce_objectness: FocalLoss,
    anchors_list: Vec<Vec<(usize, usize)>>,
    match_grid_method: MatchGrid,
    iou_kind: IoUKind,
}

impl YoloLoss {
    pub fn forward(&self, prediction: &YoloOutput, target_bboxes: Vec<Vec<RatioBBox>>) -> Tensor {
        let batch_size = prediction.batch_size();

        // match target bboxes and grids, and group them by detector cells
        // indexed by (layer_index, grid_row, grid_col)
        let matched_grids = self.match_grids(prediction, &target_bboxes);
        let grouped_target_bboxes: HashMap<(usize, i64, i64), Vec<_>> = matched_grids
            .iter()
            .flat_map(|args| {
                let (batch_index, layer_index, ref target_bbox, ref target_grids) = *args;

                target_grids
                    .iter()
                    .cloned()
                    .map(move |(grid_row, grid_col)| {
                        let key = (layer_index, grid_row, grid_col);
                        let value = (batch_index, target_bbox);
                        (key, value)
                    })
            })
            .into_group_map();

        // index detections by (layer_index, grid_row, grid_col)
        let detections: HashMap<(usize, i64, i64), _> = prediction
            .detections
            .iter()
            .map(|detection| {
                let DetectionIndex {
                    layer_index,
                    grid_row,
                    grid_col,
                    ..
                } = detection.index;

                let index = (layer_index, grid_row, grid_col);
                (index, detection)
            })
            .collect();

        // pair up targets and predictions
        let (pred_positions, pred_sizes, pred_objectnesses, pred_classifications, targets) = {
            let init_state = (vec![], vec![], vec![], vec![], vec![]);

            let final_state = grouped_target_bboxes
                .into_iter()
                .map(|(index, targets)| {
                    let detection = &detections[&index];
                    (detection, targets)
                })
                .fold(
                    init_state,
                    |mut state, (detection, targets_with_batch_indexes)| {
                        // extract batched predictions
                        let Detection {
                            position,
                            size,
                            objectness,
                            classification,
                            ..
                        } = detection;
                        let (batch_size, _) = position.size2().unwrap();

                        // select predictions by batch indexes
                        let (batch_indexes, targets) = targets_with_batch_indexes.into_iter().fold(
                            (vec![], vec![]),
                            |mut state, (batch_index, target)| {
                                let (batch_indexes, targets) = &mut state;
                                batch_indexes.push(batch_index as i64);
                                targets.push(target);
                                state
                            },
                        );
                        let batch_selector = Tensor::of_slice(&batch_indexes);

                        let selected_position = position.gather(
                            0,
                            &batch_selector.view([-1, 1]).expand_as(&position),
                            false,
                        );
                        let selected_size = position.gather(
                            0,
                            &batch_selector.view([-1, 1]).expand_as(&size),
                            false,
                        );
                        let selected_objectness = position.gather(
                            0,
                            &batch_selector.view([-1, 1]).expand_as(&objectness),
                            false,
                        );
                        let selected_classification = position.gather(
                            0,
                            &batch_selector.view([-1, 1]).expand_as(&classification),
                            false,
                        );

                        let (
                            gathered_positions,
                            gathered_sizes,
                            gathered_objectnesses,
                            gathered_classifications,
                            gathered_targets,
                        ) = &mut state;

                        gathered_positions.push(selected_position);
                        gathered_sizes.push(selected_size);
                        gathered_objectnesses.push(selected_objectness);
                        gathered_classifications.push(selected_classification);
                        gathered_targets.extend(targets);

                        state
                    },
                );

            let (
                gathered_positions,
                gathered_sizes,
                gathered_objectnesses,
                gathered_classifications,
                targets,
            ) = final_state;

            let positions = Tensor::cat(&gathered_positions, 0);
            let sizes = Tensor::cat(&gathered_sizes, 0);
            let objectnesses = Tensor::cat(&gathered_objectnesses, 0);
            let classifications = Tensor::cat(&gathered_classifications, 0);

            (positions, sizes, objectnesses, classifications, targets)
        };

        // construct target tensors
        let (target_positions, target_sizes) = {
            let init_state = (vec![], vec![], vec![], vec![], vec![]);

            let final_state = targets.iter().fold(init_state, |mut state, bbox| {
                let (cy_vec, cx_vec, h_vec, w_vec, category_id_vec) = &mut state;

                let BBox {
                    cycxhw: [cy, cx, h, w],
                    category_id,
                } = *bbox;

                cy_vec.push(cy.raw());
                cx_vec.push(cx.raw());
                h_vec.push(h.raw());
                w_vec.push(w.raw());
                category_id_vec.push(category_id);

                state
            });

            let (cy_vec, cx_vec, h_vec, w_vec, category_id_vec) = final_state;

            let target_cy = Tensor::of_slice(&cy_vec).view([-1, 1]);
            let target_cx = Tensor::of_slice(&cx_vec).view([-1, 1]);
            let target_h = Tensor::of_slice(&h_vec).view([-1, 1]);
            let target_w = Tensor::of_slice(&w_vec).view([-1, 1]);

            let target_positions = Tensor::cat(&[&target_cy, &target_cx], 1);
            let target_sizes = Tensor::cat(&[&target_h, &target_w], 1);

            (target_positions, target_sizes)
        };

        let iou_loss = self.iou_loss(
            &pred_positions,
            &pred_sizes,
            &target_positions,
            &target_sizes,
        );
        todo!();
    }

    fn iou_loss(
        &self,
        pred_positions: &Tensor,
        pred_sizes: &Tensor,
        target_positions: &Tensor,
        target_sizes: &Tensor,
    ) -> Tensor {
        // unpack parameters
        let pred_cy = pred_positions.i((.., 0));
        let pred_cx = pred_positions.i((.., 1));
        let pred_h = pred_sizes.i((.., 0));
        let pred_w = pred_sizes.i((.., 1));
        let pred_t = &pred_cy - &pred_h / 2.0;
        let pred_b = &pred_cy + &pred_h / 2.0;
        let pred_l = &pred_cx - &pred_w / 2.0;
        let pred_r = &pred_cx + &pred_w / 2.0;
        let pred_area = &pred_h * &pred_w;

        let target_cy = target_positions.i((.., 0));
        let target_cx = target_positions.i((.., 1));
        let target_h = target_sizes.i((.., 0));
        let target_w = target_sizes.i((.., 1));
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
        let union_area = &pred_area + &target_area - &intersect_area;
        let iou = &intersect_area / &union_area; // TODO: plus epsilon

        let loss = match self.iou_kind {
            IoUKind::IoU => iou,
            IoUKind::GIoU => {
                let closure_t = pred_t.min1(&target_t);
                let closure_l = pred_l.min1(&target_l);
                let closure_b = pred_b.max1(&target_b);
                let closure_r = pred_r.max1(&target_r);
                let closure_h = &closure_b - &closure_t;
                let closure_w = &closure_r - &closure_l;
                let closure_area = &closure_h * &closure_w + 1e-16;
                &iou - (&closure_area - &union_area) / &closure_area
            }
            IoUKind::CIoU => {
                todo!();
            }
            IoUKind::DIoU => {
                todo!();
            }
        };

        todo!();
    }

    /// Match target bboxes with grids.
    ///
    /// It returns the tuple (batch_index, layer_index, target_bbox, target_grids).
    fn match_grids(
        &self,
        prediction: &YoloOutput,
        target_bboxes: &Vec<Vec<RatioBBox>>,
    ) -> Vec<(usize, usize, BBox, Vec<(i64, i64)>)> {
        let image_height = prediction.image_height;
        let image_width = prediction.image_width;
        let feature_sizes = &prediction.feature_sizes;

        let bbox_iter = target_bboxes
            .iter()
            .enumerate()
            .flat_map(|(batch_index, bboxes)| {
                bboxes
                    .iter()
                    .map(move |ratio_bbox| (batch_index, ratio_bbox))
            });
        let feature_size_iter = feature_sizes.iter().cloned().enumerate();

        let targets: Vec<_> = iproduct!(bbox_iter, feature_size_iter)
            .map(
                |((batch_index, ratio_bbox), (layer_index, (height, width)))| {
                    let target_bbox = {
                        let height = R64::new(height as f64);
                        let width = R64::new(width as f64);
                        ratio_bbox.to_bbox(height, width)
                    };

                    let [cy, cx, h, w] = target_bbox.cycxhw;
                    let grid_row = cy.floor().raw() as i64;
                    let grid_col = cx.floor().raw() as i64;
                    debug_assert!(grid_row >= 0 && grid_col >= 0);
                    let margin_thresh = 0.5;

                    let target_grids: Vec<_> = {
                        let target_grid_iter = iter::once((grid_row, grid_col));
                        match self.match_grid_method {
                            MatchGrid::Rect2 => {
                                let top_iter = if cy < margin_thresh && grid_row > 0 {
                                    Some((grid_row - 1, grid_col))
                                } else {
                                    None
                                }
                                .into_iter();
                                let left_iter = if cx < margin_thresh && grid_col > 0 {
                                    Some((grid_row, grid_col - 1))
                                } else {
                                    None
                                }
                                .into_iter();

                                target_grid_iter.chain(top_iter).chain(left_iter).collect()
                            }
                            MatchGrid::Rect4 => {
                                let top_iter = if cy < margin_thresh && grid_row > 0 {
                                    Some((grid_row - 1, grid_col))
                                } else {
                                    None
                                }
                                .into_iter();
                                let left_iter = if cx < margin_thresh && grid_col > 0 {
                                    Some((grid_row, grid_col - 1))
                                } else {
                                    None
                                }
                                .into_iter();
                                let bottom_iter =
                                    if cy > (1.0 - margin_thresh) && grid_row <= height - 2 {
                                        Some((grid_row + 1, grid_col))
                                    } else {
                                        None
                                    }
                                    .into_iter();
                                let right_iter =
                                    if cx < (1.0 - margin_thresh) && grid_col <= width - 2 {
                                        Some((grid_row, grid_col + 1))
                                    } else {
                                        None
                                    }
                                    .into_iter();

                                target_grid_iter.chain(top_iter).chain(left_iter).collect()
                            }
                        }
                    };

                    (batch_index, layer_index, target_bbox, target_grids)
                },
            )
            .collect();

        targets
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
            bce,
            gamma,
            alpha,
            reduction,
        } = self;

        let bce_loss = self.bce.forward(input, target);
        let input_prob = input.sigmoid();
        let p_t: Tensor = target * &input_prob + (1.0 - target) * (1.0 - &input_prob);
        let alpha_factor = target * self.alpha + (1.0 - target) * (1.0 - self.alpha);
        let modulating_factor = (-&p_t + 1.0).pow(self.gamma);
        let loss: Tensor = bce_loss * alpha_factor * modulating_factor;

        match self.reduction {
            Reduction::Mean => loss.mean(Kind::Float),
            Reduction::Sum => loss.sum(Kind::Float),
            Reduction::None => loss,
            Reduction::Other(_) => unreachable!(),
        }
    }
}
