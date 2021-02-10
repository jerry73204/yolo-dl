use crate::{common::*, model::MergeDetect2DOutput, profiling::Timing};

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
            let mut timing = Timing::new("nms");

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
            timing.set_record("make_input");

            {
                let BatchPrediction { t, l, b, r, conf } = &batch_pred;

                // select entries corresponding conf >= threshold
                let mask = conf.ge(confidence_threshold.raw());
                let conf = conf.masked_select(&mask);
                let t = t.expand_as(&mask).masked_select(&mask);
                let l = l.expand_as(&mask).masked_select(&mask);
                let b = b.expand_as(&mask).masked_select(&mask);
                let r = r.expand_as(&mask).masked_select(&mask);

                debug_assert!({
                    let size = conf.size1().unwrap();
                    t.size1().unwrap() == size
                        && l.size1().unwrap() == size
                        && b.size1().unwrap() == size
                        && r.size1().unwrap() == size
                });

                let instant = Instant::now();
                let counts_per_batch_per_class = mask.count_nonzero(&[2]);

                {
                    let index_end = counts_per_batch_per_class
                        .view([-1])
                        .cumsum(0, Kind::Int64)
                        .view(counts_per_batch_per_class.size().as_slice());
                    let first_index_end = index_end.int64_value(&[0, 0]);
                    let index_start = index_end - first_index_end;
                }

                let index_end: Vec<i64> = counts_per_batch_per_class
                    .view([-1])
                    .cumsum(0, Kind::Int64)
                    .into();

                // this debug assertion is slow, enable it only when needed
                // debug_assert!({
                //     let (n_batches, n_classes, _n_instances) = mask.size3().unwrap();
                //     let mut counts =
                //         Array2::<usize>::zeros((n_batches as usize, n_classes as usize));
                //     let mask: Array3<bool> = mask.try_into_cv().unwrap();
                //     mask.indexed_iter()
                //         .for_each(|((batch, class, _instance), &selected)| {
                //             if selected {
                //                 counts[(batch, class)] += 1;
                //             }
                //         });

                //     let expected_index_end: Vec<_> = (0..n_batches)
                //         .flat_map(|batch| {
                //             (0..n_classes)
                //                 .map(|class| counts[(batch as usize, class as usize)])
                //                 .collect_vec()
                //         })
                //         .scan(0, |sum, count| {
                //             *sum += count;
                //             Some(*sum as i64)
                //         })
                //         .collect();

                //     index_end == expected_index_end
                // });

                let selected_bboxes: Vec<_> = izip!(
                    iter::once(0).chain(index_end.iter().cloned()),
                    index_end.iter().cloned(),
                )
                .map(|(start, end)| {
                    let conf = conf.narrow(0, start, end - start).view([-1, 1]);
                    let t = t.narrow(0, start, end - start).view([-1, 1]);
                    let l = l.narrow(0, start, end - start).view([-1, 1]);
                    let b = b.narrow(0, start, end - start).view([-1, 1]);
                    let r = r.narrow(0, start, end - start).view([-1, 1]);

                    let bboxes: TlbrConfTensor = TlbrConfTensorUnchecked {
                        tlbr: TlbrTensorUnchecked { t, l, b, r },
                        conf,
                    }
                    .try_into()
                    .unwrap();
                    let indexes = nms(&bboxes, iou_threshold.raw()).unwrap();
                    let selected_bboxes = bboxes.index_select(&indexes);

                    (selected_bboxes)
                })
                .collect();

                dbg!(instant.elapsed());

                // let instant = Instant::now();
                // let indexes = mask.nonzero();
                // let batch_indexes = indexes.select(1, 0);
                // let class_indexes = indexes.select(1, 1);
                // let instanc_indexes = indexes.select(1, 2);

                // let batch_vec: Vec<i64> = batch_indexes.into();
                // let class_vec: Vec<i64> = class_indexes.into();
                // dbg!(instant.elapsed());
                // let kinds: Vec<_> = izip!(batch_vec, class_vec).dedup_with_count().collect();
                // dbg!(kinds.len());
            };

            // select bboxes which confidence is above threshold
            // let (selected_pred, batch_indexes, class_indexes) = {
            //     let BatchPrediction { t, l, b, r, conf } = batch_pred;

            //     let mask = conf.ge(confidence_threshold.raw());
            //     let indexes = mask.nonzero();
            //     let batches = indexes.select(1, 0);
            //     let classes = indexes.select(1, 1);
            //     let instances = indexes.select(1, 2);

            //     let new_t = t
            //         .permute(&[0, 2, 1])
            //         .index(&[&batches, &instances])
            //         .view([-1, 1]);
            //     let new_l = l
            //         .permute(&[0, 2, 1])
            //         .index(&[&batches, &instances])
            //         .view([-1, 1]);
            //     let new_b = b
            //         .permute(&[0, 2, 1])
            //         .index(&[&batches, &instances])
            //         .view([-1, 1]);
            //     let new_r = r
            //         .permute(&[0, 2, 1])
            //         .index(&[&batches, &instances])
            //         .view([-1, 1]);
            //     let new_conf = conf.index(&[&batches, &classes, &instances]).view([-1, 1]);

            //     let bbox: TlbrConfTensor = TlbrConfTensorUnchecked {
            //         tlbr: TlbrTensorUnchecked {
            //             t: new_t,
            //             l: new_l,
            //             b: new_b,
            //             r: new_r,
            //         },
            //         conf: new_conf,
            //     }
            //     .try_into()
            //     .unwrap();

            //     (bbox, batches, classes)
            // };
            // timing.set_record("filter_by_confidence");

            // group bboxes by batch and class indexes
            // let nms_pred = {
            //     let batch_vec: Vec<i64> = batch_indexes.into();
            //     let class_vec: Vec<i64> = class_indexes.into();

            //     let nms_pred: HashMap<_, _> = izip!(batch_vec, class_vec)
            //         .enumerate()
            //         .map(|(select_index, (batch, class))| ((batch, class), select_index as i64))
            //         .into_group_map()
            //         .into_iter()
            //         .map(|((batch, class), select_indexes)| {
            //             // select bboxes of specfic batch and class
            //             let select_indexes = Tensor::of_slice(&select_indexes).to_device(device);
            //             let candidate_pred = selected_pred.index_select(&select_indexes);

            //             // run NMS
            //             let nms_indexes = nms(&candidate_pred, iou_threshold.raw()).unwrap();

            //             let nms_index = BatchClassIndex { batch, class };
            //             let nms_pred = candidate_pred.index_select(&nms_indexes);

            //             (nms_index, nms_pred)
            //         })
            //         .collect();

            //     nms_pred
            // };

            // timing.set_record("run_nms");
            timing.report();

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

    let mut timing = Timing::new("nms_algorithm");

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
    timing.set_record("convert_bbox");

    let permutation = PermD::from_sort_by_cached_key(conf_vec.as_slice(), |&conf| -r32(conf));
    timing.set_record("permute_confidence");

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
    timing.set_record("nms");

    let output = Tensor::of_slice(&keep)
        .set_requires_grad(false)
        .to_device(device);
    timing.set_record("create_output");

    timing.report();

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nms_test() -> Result<()> {
        let device = Device::Cpu;
        let n_samples = 96;

        let input: TlbrConfTensor = {
            let cy = Tensor::rand(&[n_samples, 1], (Kind::Float, device));
            let cx = Tensor::rand(&[n_samples, 1], (Kind::Float, device));
            let h = Tensor::rand(&[n_samples, 1], (Kind::Float, device));
            let w = Tensor::rand(&[n_samples, 1], (Kind::Float, device));

            let t = &cy - &h / 2.0;
            let b = &cy + &h / 2.0;
            let l = &cx - &w / 2.0;
            let r = &cx + &w / 2.0;

            TlbrConfTensorUnchecked {
                tlbr: TlbrTensorUnchecked { t, b, l, r },
                conf: Tensor::rand(&[n_samples, 1], (Kind::Float, device)),
            }
            .try_into()
            .unwrap()
        };

        let _nms_output = super::nms(&input, 0.5)?;

        Ok(())
    }
}
