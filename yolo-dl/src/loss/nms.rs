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
pub struct NmsOutput {
    pub batches: Tensor,
    pub classes: Tensor,
    pub instances: Tensor,
    pub bbox: TLBRTensor,
    pub confidence: Tensor,
}

impl NmsOutput {
    pub fn num_samples(&self) -> i64 {
        self.batches.size()[0]
    }
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
            let confidence_threshold = confidence_threshold.raw();
            let num_classes = prediction.num_classes;

            let MergeDetect2DOutput {
                cy,
                cx,
                h,
                w,
                class: class_logit,
                obj: obj_logit,
                ..
            } = prediction;

            // select bboxes which confidence is above threshold
            let (batches, classes, instances, bbox, conf) = {
                // compute confidence score
                let obj = obj_logit.sigmoid();
                let conf = obj.sigmoid() * class_logit.sigmoid();

                // compute tlbr bbox
                let t = cy - h / 2.0;
                let b = cy + h / 2.0;
                let l = cx - w / 2.0;
                let r = cx + w / 2.0;

                // filter by objectness and confidence (= obj * class)
                let obj_mask = obj.ge(confidence_threshold);
                let conf_mask = conf.ge(confidence_threshold);
                let mask = obj_mask.logical_and(&conf_mask);
                let indexes = mask.nonzero();
                let batches = indexes.select(1, 0);
                let classes = indexes.select(1, 1);
                let instances = indexes.select(1, 2);

                let new_t = t.index(&[Some(&batches), None, Some(&instances)]);
                let new_l = l.index(&[Some(&batches), None, Some(&instances)]);
                let new_b = b.index(&[Some(&batches), None, Some(&instances)]);
                let new_r = r.index(&[Some(&batches), None, Some(&instances)]);
                let new_conf = conf
                    .index(&[Some(&batches), Some(&classes), Some(&instances)])
                    .view([-1, 1]);

                let bbox: TLBRTensor = TLBRTensorUnchecked {
                    t: new_t,
                    l: new_l,
                    b: new_b,
                    r: new_r,
                }
                .try_into()
                .unwrap();

                (batches, classes, instances, bbox, new_conf)
            };

            let keep = {
                let ltrb = Tensor::cat(&[bbox.l(), bbox.t(), bbox.r(), bbox.b()], 1);
                let group = &batches * num_classes as i64 + &classes;
                tch_nms::nms_by_scores(&ltrb, &conf.view([-1]), &group, iou_threshold.raw())
                    .unwrap()
            };

            let keep_batches = batches.index(&[Some(&keep)]);
            let keep_classes = classes.index(&[Some(&keep)]);
            let keep_instances = instances.index(&[Some(&keep)]);
            let keep_bbox = bbox.index_select(&keep);
            let keep_conf = conf.index(&[Some(&keep)]);

            NmsOutput {
                batches: keep_batches,
                classes: keep_classes,
                instances: keep_instances,
                bbox: keep_bbox,
                confidence: keep_conf,
            }
        })
    }
}
