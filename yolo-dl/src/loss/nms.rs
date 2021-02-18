use crate::{common::*, model::MergeDetect2DOutput};

#[derive(Debug)]
pub struct NonMaxSuppressionInit {
    pub iou_threshold: R64,
    pub confidence_threshold: R64,
}

impl Default for NonMaxSuppressionInit {
    fn default() -> Self {
        Self {
            iou_threshold: r64(0.6),
            confidence_threshold: r64(0.1),
        }
    }
}

impl NonMaxSuppressionInit {
    pub fn build(self) -> Result<NonMaxSuppression> {
        let Self {
            iou_threshold,
            confidence_threshold,
        } = self;

        ensure!(iou_threshold >= 0.0, "iou_threshold must be non-negative");
        ensure!(
            confidence_threshold >= 0.0,
            "confidence_threshold must be non-negative"
        );

        Ok(NonMaxSuppression {
            iou_threshold,
            confidence_threshold,
        })
    }
}

#[derive(Debug, TensorLike)]
struct BatchPrediction {
    t: Tensor,
    l: Tensor,
    b: Tensor,
    r: Tensor,
    conf: Tensor,
}

#[derive(Debug, TensorLike, Getters)]
pub struct NmsOutput {
    #[get = "pub"]
    batch_indexes: Vec<i64>,
    #[get = "pub"]
    class_indexes: Vec<i64>,
    #[get = "pub"]
    bbox: TLBRTensor,
    #[get = "pub"]
    confidence: Tensor,
}

#[derive(Debug)]
pub struct NonMaxSuppression {
    iou_threshold: R64,
    confidence_threshold: R64,
}

impl NonMaxSuppression {
    pub fn forward(&self, prediction: &MergeDetect2DOutput) -> NmsOutput {
        tch::no_grad(|| {
            let Self {
                iou_threshold,
                confidence_threshold,
            } = *self;
            // let device = prediction.device();

            let batch_pred = {
                let MergeDetect2DOutput {
                    cy,
                    cx,
                    h,
                    w,
                    class,
                    obj,
                    ..
                } = prediction;

                // compute confidence score (= objectness * class_score)
                let conf = obj.sigmoid() * class.sigmoid();

                // compute tlbr bbox
                let t = cy - h / 2.0;
                let b = cy + h / 2.0;
                let l = cx - w / 2.0;
                let r = cx + w / 2.0;

                BatchPrediction { t, b, l, r, conf }
            };

            // select bboxes which confidence is above threshold
            let (batch_indexes, class_indexes, bbox, confidence) = {
                let BatchPrediction { t, l, b, r, conf } = batch_pred;

                let mask = conf.ge(confidence_threshold.raw());
                let indexes = mask.nonzero();
                let batches = indexes.select(1, 0);
                let classes = indexes.select(1, 1);
                let instances = indexes.select(1, 2);

                let new_t = t.index_opt((&batches, NONE_INDEX, &instances));
                let new_l = l.index_opt((&batches, NONE_INDEX, &instances));
                let new_b = b.index_opt((&batches, NONE_INDEX, &instances));
                let new_r = r.index_opt((&batches, NONE_INDEX, &instances));
                let new_conf = conf.index(&[&batches, &classes, &instances]).view([-1, 1]);

                let bbox: TLBRTensor = TLBRTensorUnchecked {
                    t: new_t,
                    l: new_l,
                    b: new_b,
                    r: new_r,
                }
                .try_into()
                .unwrap();

                (batches, classes, bbox, new_conf)
            };

            NmsOutput {
                batch_indexes: Vec::<i64>::from(&batch_indexes),
                class_indexes: Vec::<i64>::from(&class_indexes),
                bbox,
                confidence,
            }
        })
    }
}

// TODO: The algorithm is very slow. It deserves a fix.
// fn nms(bboxes: &TLBRTensor, conf: Tensor, iou_threshold: f64) -> Result<Tensor> {
//     let n_bboxes = bboxes.num_samples() as usize;
//     let device = bboxes.device();

//     let conf_vec = Vec::<f32>::from(conf);
//     let bboxes_vec: Vec<UnitlessCyCxHW<R64>> = bboxes.into();

//     let permutation = PermD::from_sort_by_cached_key(conf_vec.as_slice(), |&conf| -r32(conf));
//     let mut suppressed = vec![false; n_bboxes];
//     let mut keep: Vec<i64> = vec![];

//     for &li in permutation.indices().iter() {
//         if suppressed[li] {
//             continue;
//         }
//         keep.push(li as i64);
//         let lhs_bbox = &bboxes_vec[li];

//         for ri in (li + 1)..n_bboxes {
//             let rhs_bbox = &bboxes_vec[ri];

//             let iou = lhs_bbox.iou_with(&rhs_bbox);
//             if iou as f64 > iou_threshold {
//                 suppressed[ri] = true;
//             }
//         }
//     }

//     Ok(Tensor::of_slice(&keep)
//         .set_requires_grad(false)
//         .to_device(device))
// }
