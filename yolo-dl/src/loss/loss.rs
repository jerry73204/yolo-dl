//! Defines loss for training.

use super::{
    bce_with_logit_loss::BceWithLogitsLossInit,
    focal_loss::{FocalLoss, FocalLossInit},
    misc::{
        BoxMetric, MatchGrid, PredInstances, PredInstancesUnchecked, TargetInstances,
        TargetInstancesUnchecked,
    },
    pred_target_matching::{BBoxMatcher, BBoxMatcherInit, PredTargetMatching},
};
use crate::{common::*, model::MergeDetect2DOutput, profiling::Timing};

pub use yolo_loss::*;
pub use yolo_loss_output::*;

mod yolo_loss {
    use super::*;

    #[derive(Debug)]
    pub struct YoloLossInit {
        pub pos_weight: Option<Tensor>,
        pub reduction: Reduction,
        pub focal_loss_gamma: Option<f64>,
        pub match_grid_method: Option<MatchGrid>,
        pub box_metric: Option<BoxMetric>,
        pub smooth_classification_coef: Option<f64>,
        pub smooth_objectness_coef: Option<f64>,
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
                box_metric,
                smooth_classification_coef,
                smooth_objectness_coef,
                anchor_scale_thresh,
                iou_loss_weight,
                objectness_loss_weight,
                classification_loss_weight,
            } = self;

            let match_grid_method = match_grid_method.unwrap_or(MatchGrid::Rect4);
            let focal_loss_gamma = focal_loss_gamma.unwrap_or(0.0);
            let box_metric = box_metric.unwrap_or(BoxMetric::DIoU);
            let smooth_classification_coef = smooth_classification_coef.unwrap_or(0.01);
            let smooth_objectness_coef = smooth_objectness_coef.unwrap_or(0.0);
            let anchor_scale_thresh = anchor_scale_thresh.unwrap_or(4.0);
            let iou_loss_weight = iou_loss_weight.unwrap_or(0.05);
            let objectness_loss_weight = objectness_loss_weight.unwrap_or(1.0);
            let classification_loss_weight = classification_loss_weight.unwrap_or(0.58);

            ensure!(
                focal_loss_gamma >= 0.0,
                "focal_loss_gamma must be non-negative"
            );
            ensure!(
                smooth_classification_coef >= 0.0 && smooth_classification_coef <= 1.0,
                "smooth_classification_coef must be in range [0, 1]"
            );
            ensure!(
                smooth_objectness_coef >= 0.0 && smooth_objectness_coef <= 1.0,
                "smooth_objectness_coef must be in range [0, 1]"
            );
            ensure!(
                iou_loss_weight >= 0.0,
                "iou_loss_weight must be non-negative"
            );
            ensure!(
                objectness_loss_weight >= 0.0,
                "objectness_loss_weight must be non-negative"
            );
            ensure!(
                classification_loss_weight >= 0.0,
                "classification_loss_weight must be non-negative"
            );

            let bce_class = {
                let bce_loss = BceWithLogitsLossInit {
                    pos_weight: pos_weight.shallow_clone(),
                    ..BceWithLogitsLossInit::default(Reduction::None)
                }
                .build();
                FocalLossInit {
                    gamma: focal_loss_gamma,
                    ..FocalLossInit::default(reduction, move |input, target| {
                        bce_loss.forward(input, target)
                    })
                }
                .build()
            };

            let bce_objectness = {
                let bce_loss = BceWithLogitsLossInit {
                    pos_weight,
                    ..BceWithLogitsLossInit::default(Reduction::None)
                }
                .build();
                FocalLossInit {
                    gamma: focal_loss_gamma,
                    ..FocalLossInit::default(reduction, move |input, target| {
                        bce_loss.forward(input, target)
                    })
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
                box_metric,
                smooth_classification_coef,
                smooth_objectness_coef,
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
                box_metric: None,
                smooth_classification_coef: None,
                smooth_objectness_coef: None,
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
        box_metric: BoxMetric,
        smooth_classification_coef: f64,
        smooth_objectness_coef: f64,
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
            let matchings = self.bbox_matcher.match_bboxes(&prediction, target);
            timing.add_event("match_target_bboxes");

            // collect selected instances
            let (pred_instances, target_instances) =
                Self::collect_instances(prediction, &matchings);
            timing.add_event("collect_instances");

            // IoU loss
            let (iou_loss, iou_score) = self.iou_loss(&pred_instances, &target_instances);
            timing.add_event("iou_loss");
            debug_assert!(!bool::from(iou_loss.isnan().any()), "NaN detected");

            // classification loss
            let classification_loss = self.classification_loss(&pred_instances, &target_instances);
            timing.add_event("classification_loss");
            debug_assert!(
                !bool::from(classification_loss.isnan().any()),
                "NaN detected"
            );

            // objectness loss
            let objectness_loss = self.objectness_loss(&prediction, &matchings, iou_score.as_ref());
            timing.add_event("objectness_loss");
            debug_assert!(!bool::from(objectness_loss.isnan().any()), "NaN detected");

            // normalize and balancing
            let total_loss = self.iou_loss_weight * &iou_loss
                + self.classification_loss_weight * &classification_loss
                + self.objectness_loss_weight * &objectness_loss;
            timing.add_event("sum_losses");

            timing.report();

            (
                YoloLossOutput {
                    total_loss,
                    iou_loss,
                    classification_loss,
                    objectness_loss,
                },
                YoloLossAuxiliary {
                    target_bboxes: matchings,
                    iou_score,
                },
            )
        }

        fn iou_loss(
            &self,
            pred: &PredInstances,
            target: &TargetInstances,
        ) -> (Tensor, Option<Tensor>) {
            let pred_cycxhw = pred.cycxhw();
            let target_cycxhw = target.cycxhw();

            let (loss, iou_scores) = match self.box_metric {
                BoxMetric::Hausdorff => {
                    let distance = pred_cycxhw.hausdorff_distance_to(&target_cycxhw);
                    (distance, None)
                }
                BoxMetric::IoU | BoxMetric::GIoU | BoxMetric::DIoU | BoxMetric::CIoU => {
                    // compute IoU
                    let iou_score = match self.box_metric {
                        BoxMetric::IoU => pred_cycxhw.iou_with(&target_cycxhw),
                        BoxMetric::GIoU => pred_cycxhw.giou_with(&target_cycxhw),
                        BoxMetric::DIoU => pred_cycxhw.diou_with(&target_cycxhw),
                        BoxMetric::CIoU => pred_cycxhw.ciou_with(&target_cycxhw),
                        _ => unreachable!(),
                    };

                    // IoU loss
                    let iou_loss = 1.0 - &iou_score;

                    (iou_loss, Some(iou_score))
                }
            };

            let loss = {
                match self.reduction {
                    Reduction::None => loss.shallow_clone(),
                    Reduction::Mean => {
                        if !loss.is_empty() {
                            loss.mean(Kind::Float)
                        } else {
                            Tensor::zeros(&[], (Kind::Float, loss.device()))
                                .set_requires_grad(false)
                        }
                    }
                    Reduction::Sum => loss.sum(Kind::Float),
                    _ => panic!("reduction {:?} is not supported", self.reduction),
                }
            };

            (loss, iou_scores)
        }

        fn classification_loss(&self, pred: &PredInstances, target: &TargetInstances) -> Tensor {
            // smooth bce values
            let pos = 1.0 - 0.5 * self.smooth_classification_coef;
            let neg = 1.0 - pos;
            let pred_class = &pred.dense_class();
            let (num_instances, num_classes) = pred_class.size2().unwrap();
            let device = pred_class.device();

            // convert sparse index to dense one-hot-like
            let target_class_dense = tch::no_grad(|| {
                let target_class_sparse = &target.sparse_class();
                let pos_values =
                    Tensor::full(&target_class_sparse.size(), pos, (Kind::Float, device));
                let target_dense =
                    pred_class
                        .full_like(neg)
                        .scatter_(1, &target_class_sparse, &pos_values);

                debug_assert!({
                    let sparse_class_array: ArrayD<i64> =
                        target.sparse_class().try_into_cv().unwrap();
                    let target_array: ArrayD<f32> = (&target_dense).try_into_cv().unwrap();
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

                target_dense
            });

            self.bce_class.forward(&pred_class, &target_class_dense)
        }

        fn objectness_loss(
            &self,
            prediction: &MergeDetect2DOutput,
            matchings: &PredTargetMatching,
            scores: Option<impl Borrow<Tensor>>,
        ) -> Tensor {
            let device = prediction.device();
            let batch_size = prediction.batch_size();
            let num_targets = matchings.0.len();
            debug_assert!(scores
                .as_ref()
                .map(|scores| scores.borrow().size2().unwrap() == (num_targets as i64, 1))
                .unwrap_or(true));

            let pred_objectness = &prediction.obj;
            let target_objectness = tch::no_grad(|| {
                let (batch_indexes_vec, flat_indexes_vec) = matchings
                    .0
                    .keys()
                    .map(|instance_index| {
                        let batch_index = instance_index.batch_index as i64;
                        let flat_index = prediction.instance_to_flat_index(instance_index).unwrap();

                        debug_assert!(
                            &prediction
                                .flat_to_instance_index(batch_index as usize, flat_index)
                                .unwrap()
                                == instance_index
                        );

                        (batch_index, flat_index)
                    })
                    .unzip_n_vec();

                let batch_indexes = Tensor::of_slice(&batch_indexes_vec).to_device(device);
                let flat_indexes = Tensor::of_slice(&flat_indexes_vec).to_device(device);

                let target_objectness = {
                    let mut target = pred_objectness.zeros_like();
                    let target_scores = {
                        let mut target_scores = Tensor::full(
                            &[num_targets as i64, 1],
                            1.0 - self.smooth_objectness_coef,
                            (Kind::Float, device),
                        );

                        if let Some(scores) = &scores {
                            target_scores +=
                                scores.borrow().clamp(0.0, 1.0) * self.smooth_objectness_coef;
                        }

                        let target_scores = target_scores.set_requires_grad(false);
                        target_scores
                    };

                    let _ = target.index_put_opt_(
                        (&batch_indexes, NONE_INDEX, &flat_indexes),
                        &target_scores,
                        false,
                    );
                    target
                };

                debug_assert!({
                    let (batch_size, _num_entries, num_instances) =
                        pred_objectness.size3().unwrap();
                    let target_array: ArrayD<f32> = (&target_objectness).try_into_cv().unwrap();
                    let scores_array: Option<ArrayD<f32>> =
                        scores.map(|scores| scores.borrow().try_into_cv().unwrap());
                    let mut expect_array =
                        Array3::<f32>::zeros([batch_size as usize, 1, num_instances as usize]);
                    matchings
                        .0
                        .keys()
                        .enumerate()
                        .for_each(|(index, instance_index)| {
                            let batch_index = instance_index.batch_index as i64;
                            let flat_index =
                                prediction.instance_to_flat_index(instance_index).unwrap();
                            let target_score = {
                                let mut target_score = (1.0 - self.smooth_objectness_coef) as f32;
                                if let Some(scores_array) = &scores_array {
                                    target_score += scores_array[[index, 0]].clamp(0.0, 1.0)
                                        * self.smooth_objectness_coef as f32;
                                }
                                target_score
                            };
                            expect_array[[batch_index as usize, 0, flat_index as usize]] =
                                target_score;
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
            matchings: &PredTargetMatching,
        ) -> (PredInstances, TargetInstances) {
            let mut timing = Timing::new("collect_instances");

            let device = prediction.device();
            let pred_instances: PredInstances = {
                let (batch_indexes_vec, flat_indexes_vec) = matchings
                    .0
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
                    .index_opt((&batch_indexes, NONE_INDEX, &flat_indexes));
                let cx = prediction
                    .cx
                    .index_opt((&batch_indexes, NONE_INDEX, &flat_indexes));
                let height = prediction
                    .h
                    .index_opt((&batch_indexes, NONE_INDEX, &flat_indexes));
                let width = prediction
                    .w
                    .index_opt((&batch_indexes, NONE_INDEX, &flat_indexes));
                let objectness =
                    prediction
                        .obj
                        .index_opt((&batch_indexes, NONE_INDEX, &flat_indexes));
                let classification =
                    prediction
                        .class
                        .index_opt((&batch_indexes, NONE_INDEX, &flat_indexes));

                PredInstancesUnchecked {
                    cycxhw: CyCxHWTensorUnchecked {
                        cy,
                        cx,
                        h: height,
                        w: width,
                    },
                    objectness,
                    dense_class: classification,
                }
                .try_into()
                .unwrap()
            };

            timing.add_event("build prediction instances");

            let target_instances: TargetInstances = tch::no_grad(|| {
                let (cy_vec, cx_vec, h_vec, w_vec, category_id_vec) = matchings
                    .0
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

                TargetInstancesUnchecked {
                    cycxhw: CyCxHWTensorUnchecked {
                        cy,
                        cx,
                        h: height,
                        w: width,
                    },
                    sparse_class: category_id,
                }
                .try_into()
                .unwrap()
            });

            timing.add_event("build target instances");
            timing.report();

            (pred_instances, target_instances)
        }
    }

    #[derive(Debug)]
    pub struct YoloLossAuxiliary {
        pub target_bboxes: PredTargetMatching,
        pub iou_score: Option<Tensor>,
    }
}

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
