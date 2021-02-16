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
                bboxes.iter().filter_map(move |bbox| {
                    let [_cy, _cx, h, w] = bbox.cycxhw();
                    if abs_diff_eq!(h, 0.0) || abs_diff_eq!(w, 0.0) {
                        warn!("Ignore zero-sized bounding box {:?}.", bbox);
                        return None;
                    }
                    let bbox = Arc::new(bbox.to_owned());
                    Some((batch_index, bbox))
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
                let neighbor_grid_indexes: Vec<_> = {
                    let target_bbox_grid: LabeledGridBBox<_> =
                        target_bbox.to_r64_bbox(feature_h as usize, feature_w as usize);
                    let [target_cy, target_cx, _target_h, _target_w] = target_bbox_grid.cycxhw();
                    debug_assert!(target_cy >= 0.0 && target_cx >= 0.0);

                    let target_row = target_cy.floor().raw() as i64;
                    let target_col = target_cx.floor().raw() as i64;
                    let target_cy_fract = target_cy.fract();
                    let target_cx_fract = target_cx.fract();

                    let orig_iter = iter::once((target_row, target_col));

                    match self.match_grid_method {
                        MatchGrid::Rect2 => {
                            let top = (target_cy_fract < snap_thresh && target_row >= 1)
                                .then(|| (target_row - 1, target_col));
                            let left = (target_cx_fract < snap_thresh && target_col >= 1)
                                .then(|| (target_row, target_col - 1));

                            orig_iter.chain(top).chain(left).collect()
                        }
                        MatchGrid::Rect4 => {
                            let top = (target_cy_fract < snap_thresh && target_row >= 1)
                                .then(|| (target_row - 1, target_col));
                            let left = (target_cx_fract < snap_thresh && target_col >= 1)
                                .then(|| (target_row, target_col - 1));
                            let bottom = (target_cy_fract > (1.0 - snap_thresh)
                                && target_row <= feature_h - 2)
                                .then(|| (target_row + 1, target_col));
                            let right = (target_cx_fract > (1.0 - snap_thresh)
                                && target_col <= feature_w - 2)
                                .then(|| (target_row, target_col + 1));

                            orig_iter
                                .chain(top)
                                .chain(left)
                                .chain(bottom)
                                .chain(right)
                                .collect()
                        }
                    }
                };

                debug_assert!(neighbor_grid_indexes.iter().cloned().all(|(row, col)| {
                    row >= 0 && row <= feature_h - 1 && col >= 0 && col <= feature_w - 1
                }));

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
                let [target_cy, target_cx, target_h, target_w] = target_bbox.cycxhw();

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

                        let target_h = target_h.to_f64();
                        let target_w = target_w.to_f64();
                        let anchor_h = anchor_h.to_f64();
                        let anchor_w = anchor_w.to_f64();

                        // convert ratio to float to avoid range checking
                        let is_size_bounded = target_h / anchor_h <= self.anchor_scale_thresh
                            && anchor_h / target_h <= self.anchor_scale_thresh
                            && target_w / anchor_w <= self.anchor_scale_thresh
                            && anchor_w / target_w <= self.anchor_scale_thresh;

                        is_size_bounded.then(|| anchor_index as i64)
                    })
                    .cartesian_product(neighbor_grid_indexes.into_iter())
                    .map(move |(anchor_index, (grid_row, grid_col))| {
                        let instance_index = InstanceIndex {
                            batch_index,
                            layer_index,
                            anchor_index,
                            grid_row,
                            grid_col,
                        };

                        debug_assert!({
                            let GridSize {
                                h: feature_h,
                                w: feature_w,
                                ..
                            } = prediction.info[layer_index].feature_size;
                            let target_bbox_grid: LabeledGridBBox<_> =
                                target_bbox.to_r64_bbox(feature_h as usize, feature_w as usize);
                            let [target_cy, target_cx, _target_h, _target_w] =
                                target_bbox_grid.cycxhw();

                            (target_cy - grid_row as f64).abs() <= 1.0 + snap_thresh
                                && (target_cx - grid_col as f64).abs() <= 1.0 + snap_thresh
                        });

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
