use crate::{
    common::*,
    loss::{
        inference::YoloInferenceOutput,
        misc::MatchGrid,
        pred_target_matching::{CyCxHWMatcher, CyCxHWMatcherInit},
    },
    model::MergeDetect2DOutput,
};

#[derive(Debug, Clone)]
pub struct YoloBenchmarkInit {
    pub match_grid_method: Option<MatchGrid>,
    pub anchor_scale_thresh: Option<f64>,
}

impl YoloBenchmarkInit {
    pub fn build(self) -> Result<YoloBenchmark> {
        let Self {
            match_grid_method,
            anchor_scale_thresh,
        } = self;

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

        Ok(YoloBenchmark { bbox_matcher })
    }
}

#[derive(Debug)]
pub struct YoloBenchmark {
    bbox_matcher: CyCxHWMatcher,
}

impl YoloBenchmark {
    pub fn forward(
        &self,
        prediction: &MergeDetect2DOutput,
        inference: &YoloInferenceOutput,
        target: &[Vec<RatioLabel>],
    ) -> YoloBenchmarkOutput {
        // compute objectness accuracy
        let matching = self.bbox_matcher.match_bboxes(prediction, target);

        todo!();
    }
}

#[derive(Debug, TensorLike)]
pub struct YoloBenchmarkOutput {}
