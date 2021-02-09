use super::misc::MatchGrid;
use crate::{
    common::*,
    model::{DetectionInfo, InstanceIndex, MergeDetect2DOutput},
};

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
                            h: feature_h,
                            w: feature_w,
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
                            h: anchor_h,
                            w: anchor_w,
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
                        let instance_index = InstanceIndex {
                            batch_index,
                            layer_index,
                            anchor_index: anchor_index as i64,
                            grid_row,
                            grid_col,
                        };

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
pub struct PredTargetMatching(pub HashMap<InstanceIndex, Arc<LabeledRatioBBox>>);
