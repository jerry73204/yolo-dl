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

#[derive(Debug, PartialEq, Eq, Hash, TensorLike)]
pub struct BatchClassIndex {
    pub batch: i64,
    pub class: i64,
}

#[derive(Debug)]
pub struct NonMaxSuppressionOutput(pub HashMap<BatchClassIndex, TlbrConfTensor>);

#[derive(Debug)]
pub struct NonMaxSuppression {
    iou_threshold: R64,
    confidence_threshold: R64,
}

impl NonMaxSuppression {
    pub fn forward(&self, prediction: &MergeDetect2DOutput) -> NonMaxSuppressionOutput {
        tch::no_grad(|| {
            let Self {
                iou_threshold,
                confidence_threshold,
            } = *self;
            let device = prediction.device();

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
            let (selected_pred, batch_indexes, class_indexes) = {
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

                let bbox: TlbrConfTensor = TlbrConfTensorUnchecked {
                    tlbr: TlbrTensorUnchecked {
                        t: new_t,
                        l: new_l,
                        b: new_b,
                        r: new_r,
                    },
                    conf: new_conf,
                }
                .try_into()
                .unwrap();

                (bbox, batches, classes)
            };

            // group bboxes by batch and class indexes
            let nms_pred = {
                let batch_vec: Vec<i64> = batch_indexes.into();
                let class_vec: Vec<i64> = class_indexes.into();

                let nms_pred: HashMap<_, _> = izip!(batch_vec, class_vec)
                    .enumerate()
                    .map(|(select_index, (batch, class))| ((batch, class), select_index as i64))
                    .into_group_map()
                    .into_iter()
                    .map(|((batch, class), select_indexes)| {
                        // select bboxes of specfic batch and class
                        let select_indexes = Tensor::of_slice(&select_indexes).to_device(device);
                        let candidate_pred = selected_pred.index_select(&select_indexes);

                        // run NMS
                        let nms_indexes = nms(&candidate_pred, iou_threshold.raw()).unwrap();

                        let nms_index = BatchClassIndex { batch, class };
                        let nms_pred = candidate_pred.index_select(&nms_indexes);

                        (nms_index, nms_pred)
                    })
                    .collect();

                nms_pred
            };

            NonMaxSuppressionOutput(nms_pred)
        })
    }
}

#[derive(Debug)]
struct BatchCycxhwTensorUnchecked {
    pub cy: Tensor,
    pub cx: Tensor,
    pub h: Tensor,
    pub w: Tensor,
}

// TODO: The algorithm is very slow. It deserves a fix.
fn nms(bboxes: &TlbrConfTensor, iou_threshold: f64) -> Result<Tensor> {
    struct BndBox {
        t: f32,
        l: f32,
        b: f32,
        r: f32,
    }

    impl BndBox {
        pub fn h(&self) -> f32 {
            self.b - self.t
        }

        pub fn w(&self) -> f32 {
            self.r - self.l
        }

        pub fn area(&self) -> f32 {
            self.h() * self.w()
        }

        pub fn intersection_area_with(&self, other: &Self) -> f32 {
            let max_t = self.t.max(other.t);
            let max_l = self.l.max(other.l);
            let min_b = self.b.max(other.b);
            let min_r = self.r.max(other.r);
            let h = (min_b - max_t).max(0.0);
            let w = (min_r - max_l).max(0.0);
            h * w
        }

        pub fn iou_with(&self, other: &Self) -> f32 {
            let inter_area = self.intersection_area_with(other);
            let union_area = self.area() + other.area() - inter_area + 1e-8;
            inter_area / union_area
        }
    }

    let n_bboxes = bboxes.num_samples() as usize;
    let device = bboxes.device();

    let conf_vec = Vec::<f32>::from(bboxes.conf());
    let bboxes: Vec<_> = izip!(
        Vec::<f32>::from(bboxes.tlbr().t()),
        Vec::<f32>::from(bboxes.tlbr().l()),
        Vec::<f32>::from(bboxes.tlbr().b()),
        Vec::<f32>::from(bboxes.tlbr().r()),
    )
    .map(|(t, l, b, r)| BndBox { t, l, b, r })
    .collect();

    let permutation = PermD::from_sort_by_cached_key(conf_vec.as_slice(), |&conf| -r32(conf));
    let mut suppressed = vec![false; n_bboxes];
    let mut keep: Vec<i64> = vec![];

    for &li in permutation.indices().iter() {
        if suppressed[li] {
            continue;
        }
        keep.push(li as i64);
        let lhs_bbox = &bboxes[li];

        for ri in (li + 1)..n_bboxes {
            let rhs_bbox = &bboxes[ri];

            let iou = lhs_bbox.iou_with(&rhs_bbox);
            if iou as f64 > iou_threshold {
                suppressed[ri] = true;
            }
        }
    }

    Ok(Tensor::of_slice(&keep)
        .set_requires_grad(false)
        .to_device(device))
}
