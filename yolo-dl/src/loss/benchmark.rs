use crate::{
    common::*,
    loss::{inference::YoloInferenceOutput, MatchingOutput},
};
use tch_goodies::module::MergeDetect2DOutput;

#[derive(Debug, Clone)]
pub struct YoloBenchmarkInit {
    pub iou_threshold: R64,
    pub confidence_threshold: R64,
}

impl YoloBenchmarkInit {
    pub fn build(self) -> Result<YoloBenchmark> {
        let Self {
            iou_threshold,
            confidence_threshold,
        } = self;

        Ok(YoloBenchmark {
            iou_threshold,
            confidence_threshold,
        })
    }
}

#[derive(Debug)]
pub struct YoloBenchmark {
    iou_threshold: R64,
    confidence_threshold: R64,
}

impl YoloBenchmark {
    pub fn forward(
        &self,
        prediction: &MergeDetect2DOutput,
        matchings: &MatchingOutput,
        _inference: &YoloInferenceOutput,
    ) -> YoloBenchmarkOutput {
        let confidence_threshold = self.confidence_threshold.raw();

        // compute objectness benchmarks
        let (obj_accuracy, obj_recall, obj_precision) = {
            let all_mask = prediction.obj.sigmoid().ge(confidence_threshold);
            let matched_mask = matchings.pred.obj().sigmoid().ge(confidence_threshold);

            let all_count = all_mask.numel() as i64;
            let all_pos = i64::from(all_mask.count_nonzero(&[0, 1, 2]));
            let all_neg = all_count - all_pos;
            debug_assert!(all_count > 0);

            let matched_count = matched_mask.numel() as i64;
            let matched_pos = i64::from(matched_mask.count_nonzero(&[0, 1]));
            let matched_neg = matched_count - matched_pos;

            // let unmatched_pos = all_pos - matched_pos;
            let unmatched_neg = all_neg - matched_neg;

            let accuracy = (matched_pos + unmatched_neg) as f64 / all_count as f64;
            let recall = if matched_count != 0 {
                matched_pos as f64 / matched_count as f64
            } else {
                1.0
            };

            let precision = if all_pos != 0 {
                matched_pos as f64 / all_pos as f64
            } else {
                1.0
            };

            debug_assert!((0f64..=1f64).contains(&accuracy));
            debug_assert!((0f64..=1f64).contains(&recall));
            debug_assert!((0f64..=1f64).contains(&precision));

            (accuracy, recall, precision)
        };

        // compute classifcation benchmark
        let class_accuracy = {
            if !matchings.pred.class().is_empty() {
                let conf_mask = matchings.pred.confidence().ge(confidence_threshold);
                let (_, pred_class) = matchings.pred.class().max2(1, true);
                let class_mask = matchings.target.class().eq1(&pred_class);
                let mask = conf_mask.any1(1, true).logical_and(&class_mask);
                let accuracy = i64::from(mask.count_nonzero(&[0])) as f64 / mask.numel() as f64;
                debug_assert!((0f64..=1f64).contains(&accuracy));
                accuracy
            } else {
                1.0
            }
        };

        YoloBenchmarkOutput {
            obj_accuracy,
            obj_recall,
            obj_precision,
            class_accuracy,
        }
    }
}

#[derive(Debug, TensorLike)]
pub struct YoloBenchmarkOutput {
    pub obj_accuracy: f64,
    pub obj_precision: f64,
    pub obj_recall: f64,
    pub class_accuracy: f64,
}
