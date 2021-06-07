//! Defines loss for training.

use super::{
    bce_with_logit_loss::{BceWithLogitsLoss, BceWithLogitsLossInit},
    cross_entropy::CrossEntropyLoss,
    focal_loss::{FocalLoss, FocalLossInit},
    l2_loss::L2Loss,
    misc::{BoxMetric, MatchGrid},
    pred_target_matching::{CyCxHWMatcher, CyCxHWMatcherInit, MatchingOutput},
};
use crate::{common::*, profiling::Timing};
use tch_goodies::detection::{FlatIndex, FlatIndexTensor, MergedDenseDetection};

pub use yolo_loss::*;
pub use yolo_loss_init::*;
pub use yolo_loss_output::*;

mod yolo_loss_init {
    use super::*;

    #[derive(Debug)]
    pub struct YoloLossInit {
        pub objectness_pos_weight: Option<R64>,
        pub reduction: Reduction,
        pub focal_loss_gamma: Option<f64>,
        pub match_grid_method: Option<MatchGrid>,
        pub box_metric: Option<BoxMetric>,
        pub smooth_classification_coef: Option<f64>,
        pub smooth_objectness_coef: Option<f64>,
        pub anchor_scale_thresh: Option<f64>,
        pub iou_loss_weight: Option<f64>,
        pub objectness_loss_kind: Option<ObjectnessLossKind>,
        pub classification_loss_kind: Option<ClassificationLossKind>,
        pub objectness_loss_weight: Option<f64>,
        pub classification_loss_weight: Option<f64>,
    }

    impl YoloLossInit {
        pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>) -> Result<YoloLoss> {
            let Self {
                objectness_pos_weight,
                reduction,
                focal_loss_gamma,
                match_grid_method,
                box_metric,
                smooth_classification_coef,
                smooth_objectness_coef,
                anchor_scale_thresh,
                iou_loss_weight,
                objectness_loss_kind,
                classification_loss_kind,
                objectness_loss_weight,
                classification_loss_weight,
            } = self;
            let path = path.borrow();

            let focal_loss_gamma = focal_loss_gamma.unwrap_or(0.0);
            let box_metric = box_metric.unwrap_or(BoxMetric::DIoU);
            let smooth_classification_coef = smooth_classification_coef.unwrap_or(0.01);
            let smooth_objectness_coef = smooth_objectness_coef.unwrap_or(0.0);
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

            let class_loss = match classification_loss_kind.unwrap_or(ClassificationLossKind::Bce) {
                ClassificationLossKind::Bce => ClassificationLoss::Bce(
                    BceWithLogitsLossInit {
                        ..BceWithLogitsLossInit::default(reduction)
                    }
                    .build(path),
                ),
                ClassificationLossKind::Focal => {
                    let bce_loss = BceWithLogitsLossInit {
                        ..BceWithLogitsLossInit::default(Reduction::None)
                    }
                    .build(path);

                    ClassificationLoss::Focal(
                        FocalLossInit {
                            gamma: focal_loss_gamma,
                            ..FocalLossInit::default(reduction, move |input, target| {
                                bce_loss.forward(input, target)
                            })
                        }
                        .build(),
                    )
                }
                ClassificationLossKind::L2 => ClassificationLoss::L2(L2Loss::new(reduction)),
                ClassificationLossKind::CrossEntropy => {
                    ClassificationLoss::CrossEntropy(CrossEntropyLoss::new(false, reduction))
                }
            };

            let obj_loss = match objectness_loss_kind.unwrap_or(ObjectnessLossKind::Bce) {
                ObjectnessLossKind::Bce => ObjectnessLoss::Bce(
                    BceWithLogitsLossInit {
                        pos_weight: objectness_pos_weight
                            .map(|weight| Tensor::of_slice(&[weight.raw()])),
                        ..BceWithLogitsLossInit::default(reduction)
                    }
                    .build(path),
                ),
                ObjectnessLossKind::Focal => {
                    let bce_loss = BceWithLogitsLossInit {
                        pos_weight: objectness_pos_weight
                            .map(|weight| Tensor::of_slice(&[weight.raw()])),
                        ..BceWithLogitsLossInit::default(Reduction::None)
                    }
                    .build(path);

                    ObjectnessLoss::Focal(
                        FocalLossInit {
                            gamma: focal_loss_gamma,
                            ..FocalLossInit::default(reduction, move |input, target| {
                                bce_loss.forward(input, target)
                            })
                        }
                        .build(),
                    )
                }
                ObjectnessLossKind::L2 => ObjectnessLoss::L2(L2Loss::new(reduction)),
            };

            let bbox_matcher = {
                let mut init = CyCxHWMatcherInit::default();
                if let Some(match_grid_method) = match_grid_method {
                    init.match_grid_method = match_grid_method;
                }
                if let Some(anchor_scale_thresh) = anchor_scale_thresh {
                    init.anchor_scale_thresh = anchor_scale_thresh;
                }
                init.build()?
            };

            Ok(YoloLoss {
                reduction,
                class_loss,
                obj_loss,
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
                objectness_pos_weight: None,
                reduction: Reduction::Mean,
                focal_loss_gamma: None,
                match_grid_method: None,
                box_metric: None,
                smooth_classification_coef: None,
                smooth_objectness_coef: None,
                anchor_scale_thresh: None,
                iou_loss_weight: None,
                objectness_loss_kind: None,
                classification_loss_kind: None,
                objectness_loss_weight: None,
                classification_loss_weight: None,
            }
        }
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub enum ObjectnessLossKind {
        Bce,
        Focal,
        L2,
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub enum ClassificationLossKind {
        Bce,
        Focal,
        CrossEntropy,
        L2,
    }
}

mod yolo_loss {
    use super::*;

    #[derive(Debug)]
    pub struct YoloLoss {
        pub(super) reduction: Reduction,
        pub(super) class_loss: ClassificationLoss,
        pub(super) obj_loss: ObjectnessLoss,
        pub(super) box_metric: BoxMetric,
        pub(super) smooth_classification_coef: f64,
        pub(super) smooth_objectness_coef: f64,
        pub(super) iou_loss_weight: f64,
        pub(super) objectness_loss_weight: f64,
        pub(super) classification_loss_weight: f64,
        pub(super) bbox_matcher: CyCxHWMatcher,
    }

    impl YoloLoss {
        pub fn forward(
            &self,
            prediction: &MergedDenseDetection,
            target: &[Vec<RatioLabel>],
        ) -> (YoloLossOutput, YoloLossAuxiliary) {
            let mut timing = Timing::new("loss function");

            // match target bboxes and grids, and group them by detector cells
            // indexed by grid positions
            let matchings = self.bbox_matcher.match_bboxes(&prediction, target);
            timing.add_event("match_target_bboxes");

            // IoU loss
            let (iou_loss, iou_score) = self.iou_loss(&matchings);
            timing.add_event("iou_loss");
            debug_assert!(!bool::from(iou_loss.isnan().any()), "NaN detected");

            // classification loss
            let classification_loss = self.classification_loss(&matchings);
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
                    matchings,
                    iou_score,
                },
            )
        }

        fn iou_loss(&self, matchings: &MatchingOutput) -> (Tensor, Option<Tensor>) {
            let pred_cycxhw = matchings.pred.cycxhw();
            let target_cycxhw = matchings.target.cycxhw();

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

        fn classification_loss(&self, matchings: &MatchingOutput) -> Tensor {
            // smooth bce values
            let pos = 1.0 - 0.5 * self.smooth_classification_coef;
            let neg = 1.0 - pos;
            let pred_class = matchings.pred.class_logit();
            let (num_instances, num_classes) = pred_class.size2().unwrap();
            let device = pred_class.device();

            // convert sparse index to dense one-hot-like
            let target_class_dense = tch::no_grad(|| {
                let target_class_sparse = matchings.target.class();
                let pos_values =
                    Tensor::full(&target_class_sparse.size(), pos, (Kind::Float, device));
                let target_dense =
                    pred_class
                        .full_like(neg)
                        .scatter_(1, &target_class_sparse, &pos_values);

                debug_assert!({
                    let sparse_class_array: ArrayD<i64> =
                        matchings.target.class().try_into_cv().unwrap();
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

            self.class_loss.forward(&pred_class, &target_class_dense)
        }

        fn objectness_loss(
            &self,
            prediction: &MergedDenseDetection,
            matchings: &MatchingOutput,
            scores: Option<impl Borrow<Tensor>>,
        ) -> Tensor {
            let device = prediction.device();
            let batch_size = prediction.batch_size();
            let num_targets = matchings.num_samples();
            debug_assert!(scores
                .as_ref()
                .map(|scores| scores.borrow().size2().unwrap() == (num_targets as i64, 1))
                .unwrap_or(true));

            let pred_objectness = &prediction.obj_logit;
            let target_objectness = tch::no_grad(|| {
                let target_objectness = {
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
                    let FlatIndexTensor { batches, flats } = &matchings.pred_indexes;

                    let mut target = pred_objectness.zeros_like();
                    let _ = target.index_put_(
                        &[Some(batches), None, Some(flats)],
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

                    let flat_indexes: Vec<FlatIndex> = (&matchings.pred_indexes).into();
                    flat_indexes
                        .into_iter()
                        .enumerate()
                        .for_each(|(index, flat)| {
                            let FlatIndex {
                                batch_index,
                                flat_index,
                            } = flat;

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

            self.obj_loss.forward(
                &pred_objectness.view([batch_size, -1]),
                &target_objectness.view([batch_size, -1]),
            )
        }
    }

    #[derive(Debug)]
    pub struct YoloLossAuxiliary {
        pub matchings: MatchingOutput,
        pub iou_score: Option<Tensor>,
    }

    #[derive(Debug)]
    pub(super) enum ObjectnessLoss {
        Bce(BceWithLogitsLoss),
        Focal(FocalLoss),
        L2(L2Loss),
    }

    impl ObjectnessLoss {
        pub fn forward(&self, input: &Tensor, target: &Tensor) -> Tensor {
            match self {
                Self::Bce(loss) => loss.forward(input, target),
                Self::Focal(loss) => loss.forward(input, target),
                Self::L2(loss) => loss.forward(input, target),
            }
        }
    }

    #[derive(Debug)]
    pub(super) enum ClassificationLoss {
        Bce(BceWithLogitsLoss),
        Focal(FocalLoss),
        CrossEntropy(CrossEntropyLoss),
        L2(L2Loss),
    }

    impl ClassificationLoss {
        pub fn forward(&self, input: &Tensor, target: &Tensor) -> Tensor {
            match self {
                Self::Bce(loss) => loss.forward(input, target),
                Self::Focal(loss) => loss.forward(input, target),
                Self::CrossEntropy(loss) => loss.forward(input, target),
                Self::L2(loss) => loss.forward(input, target),
            }
        }
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
