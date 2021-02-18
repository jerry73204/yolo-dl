use super::nms::{NonMaxSuppression, NonMaxSuppressionInit};
use crate::{common::*, model::MergeDetect2DOutput};

#[derive(Debug)]
pub struct YoloBenchmarkInit {
    pub nms_iou_threshold: Option<R64>,
    pub nms_confidence_threshold: Option<R64>,
}

impl YoloBenchmarkInit {
    pub fn build(self) -> Result<YoloBenchmark> {
        let Self {
            nms_iou_threshold,
            nms_confidence_threshold,
        } = self;

        let nms = {
            let mut init = NonMaxSuppressionInit::default();
            if let Some(iou_threshold) = nms_iou_threshold {
                init.iou_threshold = iou_threshold;
            }
            if let Some(confidence_threshold) = nms_confidence_threshold {
                init.confidence_threshold = confidence_threshold;
            }
            init.build()?
        };

        Ok(YoloBenchmark { nms })
    }
}

impl Default for YoloBenchmarkInit {
    fn default() -> Self {
        Self {
            nms_iou_threshold: None,
            nms_confidence_threshold: None,
        }
    }
}

#[derive(Debug)]
pub struct YoloBenchmark {
    nms: NonMaxSuppression,
}

impl YoloBenchmark {
    pub fn forward(&self, prediction: &MergeDetect2DOutput) {
        // comput mAP
        {
            // nms
            let _nms_output = self.nms.forward(prediction);

            // group target bboxes by (batch, class)
            // let target_bboxes: HashMap<_, _> = target
            //     .iter()
            //     .enumerate()
            //     .flat_map(|(batch_index, bboxes)| {
            //         bboxes.iter().map(move |bbox| {
            //             let index = BatchClassIndex {
            //                 batch: batch_index as i64,
            //                 class: bbox.category_id as i64,
            //             };
            //             (index, bbox)
            //         })
            //     })
            //     .into_group_map()
            //     .into_iter()
            //     .map(|(index, bboxes)| {
            //         let (t_vec, l_vec, b_vec, r_vec) = bboxes
            //             .iter()
            //             .map(|bbox| {
            //                 let [t, l, b, r] = bbox.tlbr();
            //                 (
            //                     t.to_f64() as f32,
            //                     l.to_f64() as f32,
            //                     b.to_f64() as f32,
            //                     r.to_f64() as f32,
            //                 )
            //             })
            //             .unzip_n_vec();

            //         let t = Tensor::of_slice(&t_vec);
            //         let l = Tensor::of_slice(&l_vec);
            //         let b = Tensor::of_slice(&b_vec);
            //         let r = Tensor::of_slice(&r_vec);
            //         let tlbr: TLBRTensor =
            //             TLBRTensorUnchecked { t, l, b, r }.try_into().unwrap();
            //         (index, tlbr)
            //     })
            //     .collect();

            // let pred_keys: HashSet<_> = nms_output.0.keys().collect();
            // let target_keys: HashSet<_> = target_bboxes.keys().collect();
            // let common_keys = pred_keys.intersection(&target_keys);

            // // compute IoU against target bboxes
            // common_keys.into_iter().map(|index| {
            //     let pred_bboxes = &nms_output.0[index];
            //     let target_bboxes = &target_bboxes[index];
            //     todo!();
            // });
        }
    }
}
