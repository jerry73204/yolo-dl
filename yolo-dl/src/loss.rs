use crate::{
    common::*,
    model::{DetectionInfo, InstanceIndex, MergeDetect2DOutput},
    profiling::Timing,
    utils::{self, AsXY},
};

pub use average_precision::*;
pub use focal_loss::*;
pub use misc::*;
pub use multi_bce_with_logit_loss::*;
pub use pred_target_matching::*;
pub use yolo_loss::*;
pub use yolo_loss_output::*;

mod yolo_loss_output {
    use super::*;

    #[derive(Debug, TensorLike)]
    pub struct YoloLossOutput {
        pub total_loss: Tensor,
        pub iou_loss: Tensor,
        pub classification_loss: Tensor,
        pub objectness_loss: Tensor,
    }

    impl YoloLossOutput {
        pub fn weighted_mean<L>(iter: impl IntoIterator<Item = (L, f64)>) -> Result<Self>
        where
            L: Borrow<YoloLossOutput>,
        {
            let (
                total_loss_vec,
                iou_loss_vec,
                classification_loss_vec,
                objectness_loss_vec,
                weight_vec,
            ) = iter
                .into_iter()
                .map(|(loss, weight)| {
                    let YoloLossOutput {
                        total_loss,
                        iou_loss,
                        classification_loss,
                        objectness_loss,
                    } = loss.borrow().shallow_clone();

                    (
                        total_loss,
                        iou_loss,
                        classification_loss,
                        objectness_loss,
                        weight,
                    )
                })
                .unzip_n_vec();

            let weight_iter = weight_vec.iter().cloned();

            let total_loss = Tensor::f_weighted_mean_tensors(
                total_loss_vec.into_iter().zip(weight_iter.clone()),
            )?;
            let iou_loss =
                Tensor::f_weighted_mean_tensors(iou_loss_vec.into_iter().zip(weight_iter.clone()))?;
            let classification_loss = Tensor::f_weighted_mean_tensors(
                classification_loss_vec.into_iter().zip(weight_iter.clone()),
            )?;
            let objectness_loss = Tensor::f_weighted_mean_tensors(
                objectness_loss_vec.into_iter().zip(weight_iter.clone()),
            )?;

            Ok(YoloLossOutput {
                total_loss,
                iou_loss,
                classification_loss,
                objectness_loss,
            })
        }
    }
}

mod yolo_loss {
    use super::*;

    #[derive(Debug)]
    pub struct YoloLossInit {
        pub pos_weight: Option<Tensor>,
        pub reduction: Reduction,
        pub focal_loss_gamma: Option<f64>,
        pub match_grid_method: Option<MatchGrid>,
        pub iou_kind: Option<IoUKind>,
        pub iou_threshold: Option<R64>,
        pub smooth_bce_coef: Option<f64>,
        pub objectness_iou_ratio: Option<f64>,
        pub anchor_scale_thresh: Option<f64>,
        pub iou_loss_weight: Option<f64>,
        pub objectness_loss_weight: Option<f64>,
        pub classification_loss_weight: Option<f64>,
    }

    impl YoloLossInit {
        pub fn build(self) -> Result<YoloLoss> {
            let Self {
                pos_weight,
                reduction,
                focal_loss_gamma,
                match_grid_method,
                iou_kind,
                iou_threshold,
                smooth_bce_coef,
                objectness_iou_ratio,
                anchor_scale_thresh,
                iou_loss_weight,
                objectness_loss_weight,
                classification_loss_weight,
            } = self;

            let match_grid_method = match_grid_method.unwrap_or(MatchGrid::Rect4);
            let focal_loss_gamma = focal_loss_gamma.unwrap_or(0.0);
            let iou_kind = iou_kind.unwrap_or(IoUKind::DIoU);
            let iou_threshold = iou_threshold.map(|val| val.raw()).unwrap_or(0.5);
            let smooth_bce_coef = smooth_bce_coef.unwrap_or(0.01);
            let objectness_iou_ratio = objectness_iou_ratio.unwrap_or(1.0);
            let anchor_scale_thresh = anchor_scale_thresh.unwrap_or(4.0);
            let iou_loss_weight = iou_loss_weight.unwrap_or(0.05);
            let objectness_loss_weight = objectness_loss_weight.unwrap_or(1.0);
            let classification_loss_weight = classification_loss_weight.unwrap_or(0.58);

            ensure!(
                focal_loss_gamma >= 0.0,
                "focal_loss_gamma must be non-negative"
            );
            ensure!(
                smooth_bce_coef >= 0.0 && smooth_bce_coef <= 1.0,
                "smooth_bce_coef must be in range [0, 1]"
            );
            ensure!(
                objectness_iou_ratio >= 0.0 && objectness_iou_ratio <= 1.0,
                "objectness_iou_ratio must be in range [0, 1]"
            );
            ensure!(
                iou_loss_weight >= 0.0,
                "iou_loss_weight must be non-negative"
            );
            ensure!(iou_threshold >= 0.0, "iou_threshold must be non-negative");
            ensure!(
                objectness_loss_weight >= 0.0,
                "objectness_loss_weight must be non-negative"
            );
            ensure!(
                classification_loss_weight >= 0.0,
                "classification_loss_weight must be non-negative"
            );

            let bce_class = {
                let bce_loss = MultiBceWithLogitsLossInit {
                    pos_weight: pos_weight.shallow_clone(),
                    ..Default::default()
                }
                .build();
                FocalLossInit {
                    gamma: focal_loss_gamma,
                    reduction,
                    ..FocalLossInit::default(move |input, target| bce_loss.forward(input, target))
                }
                .build()
            };

            let bce_objectness = {
                let bce_loss = MultiBceWithLogitsLossInit {
                    pos_weight,
                    ..Default::default()
                }
                .build();
                FocalLossInit {
                    gamma: focal_loss_gamma,
                    reduction,
                    ..FocalLossInit::default(move |input, target| bce_loss.forward(input, target))
                }
                .build()
            };

            let bbox_matcher = BBoxMatcherInit {
                match_grid_method,
                anchor_scale_thresh,
            }
            .build()?;

            Ok(YoloLoss {
                reduction,
                bce_class,
                bce_objectness,
                iou_kind,
                iou_threshold,
                smooth_bce_coef,
                objectness_iou_ratio,
                iou_loss_weight,
                objectness_loss_weight,
                classification_loss_weight,
                bbox_matcher,
            })
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
                iou_threshold: None,
                smooth_bce_coef: None,
                objectness_iou_ratio: None,
                anchor_scale_thresh: None,
                iou_loss_weight: None,
                objectness_loss_weight: None,
                classification_loss_weight: None,
            }
        }
    }

    #[derive(Debug)]
    pub struct YoloLoss {
        reduction: Reduction,
        bce_class: FocalLoss,
        bce_objectness: FocalLoss,
        iou_kind: IoUKind,
        iou_threshold: f64,
        smooth_bce_coef: f64,
        objectness_iou_ratio: f64,
        iou_loss_weight: f64,
        objectness_loss_weight: f64,
        classification_loss_weight: f64,
        bbox_matcher: BBoxMatcher,
    }

    impl YoloLoss {
        pub fn forward(
            &self,
            prediction: &MergeDetect2DOutput,
            target: &Vec<Vec<LabeledRatioBBox>>,
        ) -> (YoloLossOutput, YoloLossAuxiliary) {
            let mut timing = Timing::new("loss function");

            // match target bboxes and grids, and group them by detector cells
            // indexed by grid positions
            // let target_bboxes = self.match_target_bboxes(&prediction, target);
            let PredTargetMatching(target_bboxes) =
                self.bbox_matcher.match_bboxes(&prediction, target);
            timing.set_record("match_target_bboxes");

            // collect selected instances
            let (pred_instances, target_instances) =
                Self::collect_instances(prediction, &target_bboxes);
            timing.set_record("collect_instances");

            // IoU loss
            let (iou_loss, iou_score) = self.iou_loss(&pred_instances, &target_instances);
            timing.set_record("iou_loss");
            debug_assert!(!bool::from(iou_loss.isnan().any()), "NaN detected");
            debug_assert!(!bool::from(iou_score.isnan().any()), "NaN detected");

            // classification loss
            let classification_loss = self.classification_loss(&pred_instances, &target_instances);
            timing.set_record("classification_loss");
            debug_assert!(
                !bool::from(classification_loss.isnan().any()),
                "NaN detected"
            );

            // objectness loss
            let objectness_loss = self.objectness_loss(&prediction, &target_bboxes, &iou_score);
            timing.set_record("objectness_loss");
            debug_assert!(!bool::from(objectness_loss.isnan().any()), "NaN detected");

            // normalize and balancing
            let total_loss = self.iou_loss_weight * &iou_loss
                + self.classification_loss_weight * &classification_loss
                + self.objectness_loss_weight * &objectness_loss;
            timing.set_record("sum_losses");

            timing.report();

            (
                YoloLossOutput {
                    total_loss,
                    iou_loss,
                    classification_loss,
                    objectness_loss,
                },
                YoloLossAuxiliary {
                    target_bboxes,
                    iou_score,
                },
            )
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
            let intersect_h = (&intersect_b - &intersect_t).clamp_min(0.0);
            let intersect_w = (&intersect_r - &intersect_l).clamp_min(0.0);
            let intersect_area = &intersect_h * &intersect_w;

            debug_assert!(
                bool::from(intersect_h.ge(0.0).all()) && bool::from(intersect_w.ge(0.0).all()),
                "negative bbox height or width detected"
            );

            // compute IoU
            let union_area = &pred_area + &target_area - &intersect_area + epsilon;
            let iou = &intersect_area / &union_area;

            let iou_score = match self.iou_kind {
                IoUKind::IoU => iou,
                _ => {
                    let outer_t = pred_t.min1(&target_t);
                    let outer_l = pred_l.min1(&target_l);
                    let outer_b = pred_b.max1(&target_b);
                    let outer_r = pred_r.max1(&target_r);
                    let outer_h = &outer_b - &outer_t;
                    let outer_w = &outer_r - &outer_l;

                    debug_assert!(
                        bool::from(outer_h.ge(0.0).all()) && bool::from(outer_w.ge(0.0).all()),
                        "negative bbox height or width detected"
                    );

                    match self.iou_kind {
                        IoUKind::GIoU => {
                            let outer_area = &outer_h * &outer_w + epsilon;
                            &iou - (&outer_area - &union_area) / &outer_area
                        }
                        _ => {
                            let diagonal_square = outer_h.pow(2.0) + outer_w.pow(2.0) + epsilon;
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
            let iou_loss = {
                let iou_loss: Tensor = 1.0 - &iou_score;
                match self.reduction {
                    Reduction::None => iou_loss.shallow_clone(),
                    Reduction::Mean => {
                        let (len, _entries) = iou_loss.size2().unwrap();
                        if len != 0 {
                            iou_loss.mean(Kind::Float)
                        } else {
                            Tensor::zeros(&[], (Kind::Float, iou_loss.device()))
                                .set_requires_grad(false)
                        }
                    }
                    Reduction::Sum => iou_loss.sum(Kind::Float),
                    _ => panic!("reduction {:?} is not supported", self.reduction),
                }
            };

            (iou_loss, iou_score)
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
            let (num_instances, num_classes) = pred_class.size2().unwrap();
            let device = pred_class.device();

            // convert sparse index to dense one-hot-like
            let target_class_dense = tch::no_grad(|| {
                let target_class_sparse = &target_instances.sparse_class;
                let pos_values =
                    Tensor::full(&target_class_sparse.size(), pos, (Kind::Float, device));
                let target =
                    pred_class
                        .full_like(neg)
                        .scatter_(1, &target_class_sparse, &pos_values);

                debug_assert!({
                    let sparse_class_array: ArrayD<i64> =
                        (&target_instances.sparse_class).try_into().unwrap();
                    let target_array: ArrayD<f32> = (&target).try_into().unwrap();
                    let expected_array = Array2::<f32>::from_shape_fn(
                        [num_instances as usize, num_classes as usize],
                        |(row, col)| {
                            let class_index = sparse_class_array[[row, 0]];
                            if class_index as usize == col {
                                pos as f32
                            } else {
                                neg as f32
                            }
                        },
                    )
                    .into_dyn();

                    let mse = (target_array - expected_array)
                        .map(|diff| diff.powi(2))
                        .mean();

                    mse.map(|mse| abs_diff_eq!(mse, 0.0)).unwrap_or(true)
                });

                target
            });

            self.bce_class.forward(&pred_class, &target_class_dense)
        }

        fn objectness_loss(
            &self,
            prediction: &MergeDetect2DOutput,
            target_bboxes: &HashMap<Arc<InstanceIndex>, Arc<LabeledRatioBBox>>,
            iou_score: &Tensor,
        ) -> Tensor {
            let device = prediction.device();
            let batch_size = prediction.batch_size();

            let pred_objectness = &prediction.obj;
            let target_objectness = tch::no_grad(|| {
                let (batch_indexes_vec, flat_indexes_vec) = target_bboxes
                    .keys()
                    .map(|instance_index| {
                        let batch_index = instance_index.batch_index as i64;
                        let flat_index = prediction.instance_to_flat_index(instance_index).unwrap();
                        (batch_index, flat_index)
                    })
                    .unzip_n_vec();

                let batch_indexes = Tensor::of_slice(&batch_indexes_vec).to_device(device);
                let flat_indexes = Tensor::of_slice(&flat_indexes_vec).to_device(device);

                let target_objectness = {
                    let target = pred_objectness.zeros_like();
                    let values = &iou_score.detach().clamp(0.0, 1.0) * self.objectness_iou_ratio
                        + (1.0 - self.objectness_iou_ratio);

                    let _ = target.permute(&[0, 2, 1]).index_put_(
                        &[&batch_indexes, &flat_indexes],
                        &values,
                        false,
                    );
                    target
                };

                debug_assert!({
                    let (batch_size, _num_entries, num_instances) =
                        pred_objectness.size3().unwrap();
                    let target_array: ArrayD<f32> = (&target_objectness).try_into().unwrap();
                    let iou_loss_array: ArrayD<f32> = iou_score.try_into().unwrap();
                    let mut expect_array =
                        Array3::<f32>::zeros([batch_size as usize, 1, num_instances as usize]);
                    target_bboxes
                        .keys()
                        .enumerate()
                        .for_each(|(index, instance_index)| {
                            let batch_index = instance_index.batch_index as i64;
                            let flat_index =
                                prediction.instance_to_flat_index(instance_index).unwrap();
                            expect_array[[batch_index as usize, 0, flat_index as usize]] =
                                iou_loss_array[[index, 0]].clamp(0.0, 1.0)
                                    * self.objectness_iou_ratio as f32
                                    + (1.0 - self.objectness_iou_ratio) as f32;
                        });

                    let mse = (target_array - expect_array)
                        .map(|val| val.powi(2))
                        .mean()
                        .unwrap();
                    abs_diff_eq!(mse, 0.0)
                });

                target_objectness
            });

            self.bce_objectness.forward(
                &pred_objectness.view([batch_size, -1]),
                &target_objectness.view([batch_size, -1]),
            )
        }

        /// Build HashSet of predictions indexed by per-grid position
        fn collect_instances(
            prediction: &MergeDetect2DOutput,
            target: &HashMap<Arc<InstanceIndex>, Arc<LabeledRatioBBox>>,
        ) -> (PredInstances, TargetInstances) {
            let mut timing = Timing::new("collect_instances");

            let device = prediction.device();
            let pred_instances = {
                let (batch_indexes_vec, flat_indexes_vec) = target
                    .keys()
                    .map(|instance_index| {
                        let batch_index = instance_index.batch_index as i64;
                        let flat_index = prediction.instance_to_flat_index(instance_index).unwrap();
                        (batch_index, flat_index)
                    })
                    .unzip_n_vec();
                let batch_indexes = Tensor::of_slice(&batch_indexes_vec).to_device(device);
                let flat_indexes = Tensor::of_slice(&flat_indexes_vec).to_device(device);

                let cy = prediction
                    .cy
                    .permute(&[0, 2, 1])
                    .index(&[&batch_indexes, &flat_indexes]);
                let cx = prediction
                    .cx
                    .permute(&[0, 2, 1])
                    .index(&[&batch_indexes, &flat_indexes]);
                let height = prediction
                    .h
                    .permute(&[0, 2, 1])
                    .index(&[&batch_indexes, &flat_indexes]);
                let width = prediction
                    .w
                    .permute(&[0, 2, 1])
                    .index(&[&batch_indexes, &flat_indexes]);
                let objectness = prediction
                    .obj
                    .permute(&[0, 2, 1])
                    .index(&[&batch_indexes, &flat_indexes]);
                let classification = prediction
                    .class
                    .permute(&[0, 2, 1])
                    .index(&[&batch_indexes, &flat_indexes]);

                PredInstances {
                    cy,
                    cx,
                    height,
                    width,
                    objectness,
                    dense_class: classification,
                }
            };

            timing.set_record("build prediction instances");

            let target_instances = tch::no_grad(|| {
                let (cy_vec, cx_vec, h_vec, w_vec, category_id_vec) = target
                    .values()
                    .map(|bbox| {
                        let [cy, cx, h, w] = bbox.bbox.cycxhw();
                        let category_id = bbox.category_id;
                        (
                            cy.to_f64() as f32,
                            cx.to_f64() as f32,
                            h.to_f64() as f32,
                            w.to_f64() as f32,
                            category_id as i64,
                        )
                    })
                    .unzip_n_vec();

                let cy = Tensor::of_slice(&cy_vec)
                    .view([-1, 1])
                    .set_requires_grad(false)
                    .to_device(device);
                let cx = Tensor::of_slice(&cx_vec)
                    .view([-1, 1])
                    .set_requires_grad(false)
                    .to_device(device);
                let height = Tensor::of_slice(&h_vec)
                    .view([-1, 1])
                    .set_requires_grad(false)
                    .to_device(device);
                let width = Tensor::of_slice(&w_vec)
                    .view([-1, 1])
                    .set_requires_grad(false)
                    .to_device(device);
                let category_id = Tensor::of_slice(&category_id_vec)
                    .view([-1, 1])
                    .set_requires_grad(false)
                    .to_device(device);

                TargetInstances {
                    cy,
                    cx,
                    height,
                    width,
                    sparse_class: category_id,
                }
            });

            timing.set_record("build target instances");
            timing.report();

            (pred_instances, target_instances)
        }
    }

    #[derive(Debug)]
    pub struct YoloLossAuxiliary {
        pub target_bboxes: HashMap<Arc<InstanceIndex>, Arc<LabeledRatioBBox>>,
        pub iou_score: Tensor,
    }
}

mod multi_bce_with_logit_loss {
    use super::*;

    #[derive(Debug)]
    pub struct MultiBceWithLogitsLossInit {
        pub weight: Option<Tensor>,
        pub pos_weight: Option<Tensor>,
        pub reduction: Reduction,
    }

    impl MultiBceWithLogitsLossInit {
        pub fn build(self) -> MultiBceWithLogitsLoss {
            let Self {
                weight,
                pos_weight,
                reduction,
            } = self;

            MultiBceWithLogitsLoss {
                weight,
                pos_weight,
                reduction,
            }
        }
    }

    impl Default for MultiBceWithLogitsLossInit {
        fn default() -> Self {
            Self {
                weight: None,
                pos_weight: None,
                reduction: Reduction::Mean,
            }
        }
    }

    #[derive(Debug)]
    pub struct MultiBceWithLogitsLoss {
        weight: Option<Tensor>,
        pos_weight: Option<Tensor>,
        reduction: Reduction,
    }

    impl MultiBceWithLogitsLoss {
        pub fn forward(&self, input: &Tensor, target: &Tensor) -> Tensor {
            // assume [batch_size, n_classes] shape
            debug_assert_eq!(
                input.size2().unwrap(),
                target.size2().unwrap(),
                "input and target tensors must have equal shape"
            );
            debug_assert!(
                bool::from(target.ge(0.0).logical_and(&target.le(1.0)).all()),
                "target values must be in range of [0.0, 1.0]"
            );

            // return zero tensor if (1) input is empty and (2) using mean reduction
            if input.is_empty() && self.reduction == Reduction::Mean {
                return Tensor::zeros(&[], (Kind::Float, input.device())).set_requires_grad(false);
            }

            input.binary_cross_entropy_with_logits(
                target,
                self.weight.as_ref(),
                self.pos_weight.as_ref(),
                self.reduction,
            )
        }
    }
}

mod focal_loss {
    use super::*;

    #[derive(Derivative)]
    #[derivative(Debug)]
    pub struct FocalLossInit<F>
    where
        F: 'static + Fn(&Tensor, &Tensor) -> Tensor + Send,
    {
        #[derivative(Debug = "ignore")]
        pub loss_fn: F,
        pub gamma: f64,
        pub alpha: f64,
        pub reduction: Reduction,
    }

    impl<F> FocalLossInit<F>
    where
        F: 'static + Fn(&Tensor, &Tensor) -> Tensor + Send,
    {
        pub fn default(loss_fn: F) -> Self {
            Self {
                loss_fn,
                gamma: 1.5,
                alpha: 0.25,
                reduction: Reduction::Mean,
            }
        }

        pub fn build(self) -> FocalLoss {
            let Self {
                loss_fn,
                gamma,
                alpha,
                reduction,
            } = self;

            FocalLoss {
                loss_fn: Box::new(loss_fn),
                gamma,
                alpha,
                reduction,
            }
        }
    }

    #[derive(Derivative)]
    #[derivative(Debug)]
    pub struct FocalLoss {
        #[derivative(Debug = "ignore")]
        loss_fn: Box<dyn Fn(&Tensor, &Tensor) -> Tensor + Send>,
        gamma: f64,
        alpha: f64,
        reduction: Reduction,
    }

    impl FocalLoss {
        pub fn forward(&self, input: &Tensor, target: &Tensor) -> Tensor {
            debug_assert_eq!(
                input.size2().unwrap(),
                target.size2().unwrap(),
                "input and target shape must be equal"
            );
            debug_assert!(
                bool::from(target.ge(0.0).logical_and(&target.le(1.0)).all()),
                "target values must be in range of [0.0, 1.0]"
            );

            // return zero tensor if (1) input is empty and (2) using mean reduction
            if input.is_empty() && self.reduction == Reduction::Mean {
                return Tensor::zeros(&[], (Kind::Float, input.device())).set_requires_grad(false);
            }

            let Self {
                ref loss_fn,
                gamma,
                alpha,
                reduction,
            } = *self;

            let orig_loss = loss_fn(input, target);
            debug_assert_eq!(
                orig_loss.size2().unwrap(),
                target.size2().unwrap(),
                "the contained loss function must not apply reduction"
            );

            let input_prob = input.sigmoid();
            let p_t: Tensor = target * &input_prob + (1.0 - target) * (1.0 - &input_prob);
            let alpha_factor = target * alpha + (1.0 - target) * (1.0 - alpha);
            let modulating_factor = (-&p_t + 1.0).pow(gamma);
            let loss: Tensor = &orig_loss * &alpha_factor * &modulating_factor;

            match reduction {
                Reduction::None => loss,
                Reduction::Sum => loss.sum(Kind::Float),
                Reduction::Mean => loss.mean(Kind::Float),
                Reduction::Other(_) => unimplemented!(),
            }
        }
    }
}

mod misc {
    use super::*;

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
}

mod pred_target_matching {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct BBoxMatcherInit {
        pub match_grid_method: MatchGrid,
        pub anchor_scale_thresh: f64,
    }

    impl BBoxMatcherInit {
        pub fn build(self) -> Result<BBoxMatcher> {
            let Self {
                match_grid_method,
                anchor_scale_thresh,
            } = self;
            ensure!(
                anchor_scale_thresh.is_finite(),
                "anchor_scale_thresh must be a finite number"
            );
            ensure!(
                anchor_scale_thresh >= 1.0,
                "anchor_scale_thresh must be greater than or equal to 1"
            );
            Ok(BBoxMatcher {
                match_grid_method,
                anchor_scale_thresh,
            })
        }
    }

    #[derive(Debug, Clone)]
    pub struct BBoxMatcher {
        match_grid_method: MatchGrid,
        anchor_scale_thresh: f64,
    }

    impl BBoxMatcher {
        /// Match predicted and target bboxes.
        pub fn match_bboxes(
            &self,
            prediction: &MergeDetect2DOutput,
            target: &Vec<Vec<LabeledRatioBBox>>,
        ) -> PredTargetMatching {
            let snap_thresh = 0.5;

            let target_bboxes: HashMap<_, _> = target
            .iter()
            .enumerate()
            // filter out small bboxes
            .flat_map(|(batch_index, bboxes)| {
                bboxes.iter().map(move |bbox| {
                    let [_cy, _cx, h, w] = bbox.cycxhw();
                    if abs_diff_eq!(h, 0.0) || abs_diff_eq!(w, 0.0) {
                        warn!(
                            "The bounding box {:?} is too small. It may cause division by zero error.",
                            bbox
                        );
                    }
                    let bbox = Arc::new(bbox.to_owned());
                    (batch_index, bbox)
                })
            })
            // pair up per target bbox and per prediction feature
            .cartesian_product(prediction.info.iter().enumerate())
            // generate neighbor bboxes
            .map(|args| {
                // unpack variables
                let ((batch_index, target_bbox), (layer_index, layer)) = args;
                let DetectionInfo {
                    feature_size:
                        GridSize {
                            height: feature_h,
                            width: feature_w,
                            ..
                        },
                    ref anchors,
                    ..
                } = *layer;

                // collect neighbor grid indexes
                let neighbor_grid_indexes = {
                    let target_bbox_grid: LabeledGridBBox<_> = target_bbox
                        .to_r64_bbox(feature_h as usize, feature_w as usize);
                    let [target_cy, target_cx, _target_h, _target_w] = target_bbox_grid.cycxhw();
                    let target_row = target_cy.floor().raw() as i64;
                    let target_col = target_cx.floor().raw() as i64;
                    debug_assert!(target_row >= 0 && target_col >= 0);

                    let grid_indexes: Vec<_> = {
                        let orig_iter = iter::once((target_row, target_col));
                        match self.match_grid_method {
                            MatchGrid::Rect2 => {
                                let top_iter = if target_cy % 1.0 < snap_thresh && target_row > 0 {
                                    Some((target_row - 1, target_col))
                                } else {
                                    None
                                }
                                .into_iter();
                                let left_iter = if target_cx < snap_thresh && target_col > 0 {
                                    Some((target_row, target_col - 1))
                                } else {
                                    None
                                }
                                .into_iter();

                                orig_iter.chain(top_iter).chain(left_iter).collect()
                            }
                            MatchGrid::Rect4 => {
                                let top_iter = if target_cy % 1.0 < snap_thresh && target_row > 0 {
                                    Some((target_row - 1, target_col))
                                } else {
                                    None
                                }
                                .into_iter();
                                let left_iter = if target_cx < snap_thresh && target_col > 0 {
                                    Some((target_row, target_col - 1))
                                } else {
                                    None
                                }
                                .into_iter();
                                let bottom_iter = if target_cy % 1.0 > (1.0 - snap_thresh)
                                    && target_row <= feature_h - 2
                                {
                                    Some((target_row + 1, target_col))
                                } else {
                                    None
                                }
                                .into_iter();
                                let right_iter = if target_cx % 1.0 > (1.0 - snap_thresh)
                                    && target_col <= feature_w - 2
                                {
                                    Some((target_row, target_col + 1))
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

                (
                    batch_index,
                    layer_index,
                    target_bbox,
                    neighbor_grid_indexes,
                    anchors,
                )
            })
            // pair each target bbox with each anchor
            .flat_map(|args| {
                let (batch_index, layer_index, target_bbox, neighbor_grid_indexes, anchors) = args;
                let [_target_cy, _target_cx, target_h, target_w] = target_bbox.cycxhw();

                // pair up anchors and neighbor grid indexes
                anchors
                    .iter()
                    .cloned()
                    .enumerate()
                    .filter_map(move |(anchor_index, anchor_size)| {
                        // filter by anchor sizes
                        let RatioSize {
                            height: anchor_h,
                            width: anchor_w,
                            ..
                        } = anchor_size;

                        // convert ratio to float to avoid range checking
                        if target_h.to_f64() / anchor_h.to_f64() <= self.anchor_scale_thresh
                            && anchor_h.to_f64() / target_h.to_f64() <= self.anchor_scale_thresh
                            && target_w.to_f64() / anchor_w.to_f64() <= self.anchor_scale_thresh
                            && anchor_w.to_f64() / target_w.to_f64() <= self.anchor_scale_thresh
                        {
                            Some(anchor_index)
                        } else {
                            None
                        }
                    })
                    .cartesian_product(neighbor_grid_indexes.into_iter())
                    .map(move |(anchor_index, (grid_row, grid_col))| {
                        let instance_index = Arc::new(InstanceIndex {
                            batch_index,
                            layer_index,
                            anchor_index: anchor_index as i64,
                            grid_row,
                            grid_col,
                        });

                        (instance_index, target_bbox.clone())
                    })
            })
            // group up target bboxes which snap to the same grid position
            .into_group_map()
            .into_iter()
            // build 1-to-1 correspondence of target bboxes and instance indexes
            .scan(
                rand::thread_rng(),
                |rng, (instance_index, target_bboxes)| {
                    let target_bbox = target_bboxes.choose(rng).unwrap().clone();
                    Some((instance_index, target_bbox))
                },
            )
            .collect();

            PredTargetMatching(target_bboxes)
        }
    }

    #[derive(Debug, Clone)]
    pub struct PredTargetMatching(pub HashMap<Arc<InstanceIndex>, Arc<LabeledRatioBBox>>);
}

mod average_precision {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct DetectionForAp<G>
    where
        G: Eq + Ord,
    {
        pub ground_truth: Option<G>,
        pub confidence: R64,
        pub iou: R64,
    }

    #[derive(Debug, Clone)]
    pub struct PrecRec<T>
    where
        T: Copy,
    {
        pub precision: T,
        pub recall: T,
    }

    impl<T> AsXY<T, T> for PrecRec<T>
    where
        T: Copy,
    {
        fn x(&self) -> T {
            self.recall
        }

        fn y(&self) -> T {
            self.precision
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub enum IntegralMethod {
        Continuous,
        Interpolation(usize),
    }

    #[derive(Debug)]
    pub struct ApCalculator {
        integral_method: IntegralMethod,
    }

    impl ApCalculator {
        pub fn new(integral_method: IntegralMethod) -> Result<Self> {
            if let IntegralMethod::Interpolation(n_points) = integral_method {
                ensure!(
                    n_points >= 1,
                    "invalid number of interpolated points {}",
                    n_points
                );
            }

            Ok(Self { integral_method })
        }

        /// Compute average precision from a precision/recall curve.
        ///
        /// The input precision/recall list must be ordered by non-increasing recall.
        pub fn compute_by_prec_rec(&self, sorted_prec_rec: &[PrecRec<R64>]) -> R64 {
            // compute precision envelope
            let enveloped = {
                let max_recall = sorted_prec_rec.last().unwrap().recall;
                let first = PrecRec {
                    precision: r64(0.0),
                    recall: r64(0.0),
                };
                let last = PrecRec {
                    precision: r64(0.0),
                    recall: (max_recall + 1e-3).min(r64(1.0)),
                };

                // append/prepend sentinel values
                let iter = {
                    iter::once(&first)
                        .chain(sorted_prec_rec.iter())
                        .chain(iter::once(&last))
                };

                let mut list: Vec<_> = iter
                    .rev()
                    .scan(r64(0.0), |max_prec: &mut R64, prec_rec: &PrecRec<R64>| {
                        let PrecRec { precision, recall } = *prec_rec;
                        *max_prec = (*max_prec).max(precision);
                        Some(PrecRec { recall, precision })
                    })
                    .collect();
                list.reverse();
                list
            };

            // compute ap
            match self.integral_method {
                IntegralMethod::Interpolation(n_points) => {
                    let points_iter =
                        (0..n_points).map(|index| r64(index as f64 / n_points as f64));
                    let interpolated: Vec<_> = utils::interpolate_slice(points_iter, &enveloped)
                        .into_iter()
                        .map(|(recall, precision)| PrecRec { recall, precision })
                        .collect();
                    let ap = utils::trapz(&interpolated);
                    ap
                }
                IntegralMethod::Continuous => {
                    todo!();
                }
            }
        }

        pub fn compute_by_detections<'a, I, G>(
            &self,
            rows: I,
            num_ground_truth: usize,
            iou_thresh: R64,
        ) -> R64
        where
            I: IntoIterator<Item = &'a DetectionForAp<G>>,
            G: Eq + Ord + 'a,
        {
            // sort by ground truth and decreasing IoU
            let mut rows: Vec<_> = rows.into_iter().collect();
            rows.sort_by_cached_key(|row| (&row.ground_truth, -row.iou));

            // Mark true positive (tp) for every first detection with IoU >= threshold
            // for every ground truth
            let mut rows: Vec<_> = rows
                .into_iter()
                .scan(None, |prev_ground_truth, row| {
                    let is_same_ground_truth = prev_ground_truth
                        .zip(row.ground_truth.as_ref())
                        .map_or(false, |(prev, curr)| prev == curr);

                    let is_tp = if is_same_ground_truth {
                        false
                    } else {
                        row.iou >= iou_thresh
                    };

                    *prev_ground_truth = row.ground_truth.as_ref();

                    Some((row, is_tp))
                })
                .collect();

            // sort by decreasing confidence
            rows.sort_by_key(|(row, _is_tp)| -row.confidence);

            // compute precision and recall, it is ordered by increasing recall automatically
            let prec_rec: Vec<_> = rows
                .into_iter()
                .scan((0, 0), |(acc_tp, acc_fp), (_row, is_tp)| {
                    if is_tp {
                        *acc_tp += 1;
                    } else {
                        *acc_fp += 1;
                    }
                    let prec_rec = {
                        let acc_tp = r64(*acc_tp as f64);
                        let acc_fp = r64(*acc_fp as f64);
                        PrecRec {
                            precision: acc_tp / (acc_tp + acc_fp),
                            recall: acc_tp / num_ground_truth as f64,
                        }
                    };
                    Some(prec_rec)
                })
                .collect();

            // compute ap
            let ap = self.compute_by_prec_rec(&prec_rec);
            ap
        }
    }

    #[derive(Debug)]
    pub struct CocoMapCalculator {
        calculator: ApCalculator,
    }

    impl CocoMapCalculator {
        pub fn new() -> Result<Self> {
            Ok(Self {
                calculator: ApCalculator::new(IntegralMethod::Interpolation(101))?,
            })
        }

        pub fn compute_mean_ap<I, G>(
            &self,
            rows: &[DetectionForAp<G>],
            num_ground_truth: usize,
            iou_thresholds: impl IntoIterator<Item = R64>,
        ) -> R64
        where
            G: Eq + Ord,
        {
            let sum_ap: R64 = iou_thresholds
                .into_iter()
                .map(|iou_thresh| {
                    let ap = self.calculator.compute_by_detections(
                        rows.iter(),
                        num_ground_truth,
                        iou_thresh,
                    );
                    ap
                })
                .sum();
            let mean_ap = sum_ap / rows.len() as f64;
            mean_ap
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn focal_loss() -> Result<()> {
        let mut rng = rand::thread_rng();
        let device = Device::cuda_if_available();

        let n_batch = 32;
        let n_class = rng.gen_range(1..10);

        let vs = nn::VarStore::new(device);
        let root = vs.root();
        let loss_fn = {
            let bce = MultiBceWithLogitsLossInit {
                reduction: Reduction::None,
                ..Default::default()
            }
            .build();
            let focal =
                FocalLossInit::default(move |input, target| bce.forward(input, target)).build();
            focal
        };

        let input = root.randn("input", &[n_batch, n_class], 0.0, 100.0);
        let target = Tensor::randn(&[n_batch, n_class], (Kind::Float, device))
            .ge(0.5)
            .to_kind(Kind::Float)
            .set_requires_grad(false);

        let mut optimizer = nn::Adam::default().build(&vs, 1.0)?;

        for _ in 0..1000 {
            let loss = loss_fn.forward(&input, &target);
            optimizer.backward_step(&loss);
        }

        optimizer.set_lr(0.1);

        for _ in 0..10000 {
            let loss = loss_fn.forward(&input, &target);
            optimizer.backward_step(&loss);
        }

        ensure!(
            bool::from((input.sigmoid() - &target).abs().le(1e-3).all()),
            "the loss does not coverage"
        );
        Ok(())
    }
}
