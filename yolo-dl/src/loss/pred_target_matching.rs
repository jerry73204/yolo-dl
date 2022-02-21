use super::misc::MatchGrid;
use crate::{common::*, label::RatioLabel};
use bbox::{prelude::*, CyCxHW};
use tch_goodies::{
    detection::{
        DetectionInfo, FlatIndexTensor, InstanceIndex, InstanceIndexTensor, MergedDenseDetection,
    },
    CyCxHWTensorUnchecked, LabelTensor, LabelTensorUnchecked, ObjectDetectionTensor, Pixel,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CyCxHWMatcherInit {
    pub match_grid_method: MatchGrid,
    pub anchor_scale_thresh: R64,
}

impl Default for CyCxHWMatcherInit {
    fn default() -> Self {
        Self {
            match_grid_method: MatchGrid::Rect4,
            anchor_scale_thresh: r64(4.0),
        }
    }
}

impl CyCxHWMatcherInit {
    pub fn build(self) -> Result<CyCxHWMatcher> {
        let Self {
            match_grid_method,
            anchor_scale_thresh,
        } = self;
        ensure!(
            anchor_scale_thresh >= 1.0,
            "anchor_scale_thresh must be greater than or equal to 1"
        );
        Ok(CyCxHWMatcher {
            match_grid_method,
            anchor_scale_thresh: anchor_scale_thresh.raw(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct CyCxHWMatcher {
    match_grid_method: MatchGrid,
    anchor_scale_thresh: f64,
}

impl CyCxHWMatcher {
    /// Match predicted and target bboxes.
    pub fn match_bboxes(
        &self,
        prediction: &MergedDenseDetection,
        target: &[Vec<RatioLabel>],
    ) -> MatchingOutput {
        let snap_thresh = 0.5;

        let target_bboxes: HashMap<InstanceIndex, Arc<RatioLabel>> = target
            .iter()
            .enumerate()
            // filter out small bboxes
            .flat_map(|(batch_index, bboxes)| {
                bboxes.iter().filter_map(move |bbox| {
                    let h = bbox.rect.h().raw();
                    let w = bbox.rect.w().raw();
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
                    ref feature_size,
                    ref anchors,
                    ..
                } = *layer;

                // collect neighbor grid indexes
                let neighbor_grid_indexes: Vec<_> = {
                    let target_bbox_grid: Pixel<CyCxHW<R64>> = {
                        let [h, w] = feature_size.cast::<R64>().hw();
                        Pixel(target_bbox.rect.scale_hw(h, w))
                    };
                    let cy = target_bbox_grid.cy();
                    let cx = target_bbox_grid.cx();
                    debug_assert!(cy >= 0.0 && cx >= 0.0);

                    let row = cy.floor().raw() as i64;
                    let col = cx.floor().raw() as i64;
                    let cy_fract = cy.fract();
                    let cx_fract = cx.fract();

                    let pos_c = iter::once((row, col));
                    let pos_t = (cy_fract < snap_thresh).then(|| (row - 1, col));
                    let pos_l = (cx_fract < snap_thresh).then(|| (row, col - 1));
                    let pos_b = (self.match_grid_method == MatchGrid::Rect4
                        && cy_fract > 1.0 - snap_thresh)
                        .then(|| (row + 1, col));
                    let pos_r = (self.match_grid_method == MatchGrid::Rect4
                        && cx_fract > 1.0 - snap_thresh)
                        .then(|| (row, col + 1));

                    chain!(pos_c, pos_t, pos_l, pos_b, pos_r)
                        .filter(|(row, col)| {
                            (0..feature_size.h()).contains(row)
                                && (0..feature_size.w()).contains(col)
                        })
                        .collect()
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
                let target_h = target_bbox.rect.h().raw();
                let target_w = target_bbox.rect.w().raw();

                // pair up anchors and neighbor grid indexes
                anchors
                    .iter()
                    .cloned()
                    .enumerate()
                    .filter_map(move |(anchor_index, anchor_size)| {
                        // filter by anchor sizes
                        let [anchor_h, anchor_w] = anchor_size.cast::<f64>().hw();

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
                            batch_index: batch_index as i64,
                            layer_index: layer_index as i64,
                            anchor_index,
                            grid_row,
                            grid_col,
                        };

                        debug_assert!({
                            let feature_size = &prediction.info[layer_index].feature_size;
                            let target_bbox_grid: Pixel<CyCxHW<R64>> = {
                                let [h, w] = feature_size.cast::<R64>().hw();
                                Pixel(target_bbox.rect.scale_hw(h, w))
                            };
                            let target_cy = target_bbox_grid.cy();
                            let target_cx = target_bbox_grid.cx();

                            (target_cy - grid_row as f64).abs() <= 1.0 + snap_thresh
                                && (target_cx - grid_col as f64).abs() <= 1.0 + snap_thresh
                        });

                        (instance_index, target_bbox.clone())
                    })
            })
            // match each grid per layer per anchor to at most one target bbox
            .fold(
                HashMap::new(),
                |mut matchings, (instance_index, target_bbox)| {
                    let InstanceIndex {
                        layer_index,
                        grid_row,
                        grid_col,
                        ..
                    } = instance_index;

                    match matchings.entry(instance_index) {
                        hash_map::Entry::Occupied(mut entry) => {
                            let orig_bbox = entry.get_mut();
                            let feature_size = &prediction.info[layer_index as usize].feature_size;
                            let pred_cy = (grid_row as f64 + 0.5) / feature_size.h() as f64;
                            let pred_cx = (grid_col as f64 + 0.5) / feature_size.w() as f64;

                            let dist_orig = {
                                (orig_bbox.rect.cy() - pred_cy).powi(2)
                                    + (orig_bbox.rect.cx() - pred_cx).powi(2)
                            };
                            let dist_new = {
                                (target_bbox.rect.cy() - pred_cy).powi(2)
                                    + (target_bbox.rect.cx() - pred_cx).powi(2)
                            };

                            if dist_new < dist_orig {
                                let _ = mem::replace(orig_bbox, target_bbox);
                            }
                        }
                        hash_map::Entry::Vacant(entry) => {
                            entry.insert(target_bbox);
                        }
                    }

                    matchings
                },
            );

        debug_assert!(target_bboxes.iter().all(|(instance_index, target_bbox)| {
            let InstanceIndex {
                layer_index,
                grid_row,
                grid_col,
                ..
            } = *instance_index;
            let feature_size = &prediction.info[layer_index as usize].feature_size;
            let target_bbox_grid: Pixel<CyCxHW<f64>> = {
                let [h, w] = feature_size.cast::<f64>().hw();
                Pixel(target_bbox.rect.cast::<f64>().scale_hw(h, w))
            };
            let target_cy = target_bbox_grid.cy();
            let target_cx = target_bbox_grid.cx();

            (target_cy - grid_row as f64).abs() <= 1.0 + snap_thresh
                && (target_cx - grid_col as f64).abs() <= 1.0 + snap_thresh
        }));

        let device = prediction.device();
        let pred_indexes = prediction
            .instances_to_flats(&InstanceIndexTensor::from_iter(target_bboxes.keys()))
            .unwrap()
            .to_device(device);
        let pred = prediction.index_by_flats(&pred_indexes);
        let target = {
            let (cy, cx, h, w, class) = target_bboxes
                .values()
                .map(|label| {
                    let [cy, cx, h, w] = label.rect.cast::<f32>().cycxhw();
                    let class = label.class as i64;
                    (cy, cx, h, w, class)
                })
                .unzip_n_vec();

            let tensor: LabelTensor = LabelTensorUnchecked {
                cycxhw: CyCxHWTensorUnchecked {
                    cy: Tensor::of_slice(&cy),
                    cx: Tensor::of_slice(&cx),
                    h: Tensor::of_slice(&h),
                    w: Tensor::of_slice(&w),
                },
                class: Tensor::of_slice(&class),
            }
            .try_into()
            .unwrap();

            // LabelTensor::from_iter(target_bboxes.values().map(|label| &**label)).to_device(device)
            tensor
        };

        MatchingOutput {
            pred_indexes,
            pred,
            target,
        }
    }
}

#[derive(Debug, TensorLike)]
pub struct MatchingOutput {
    pub pred_indexes: FlatIndexTensor,
    pub pred: ObjectDetectionTensor,
    pub target: LabelTensor,
}

impl MatchingOutput {
    pub fn num_samples(&self) -> i64 {
        self.pred_indexes.num_samples()
    }

    pub fn cat_mini_batches<T>(iter: impl IntoIterator<Item = (T, i64)>) -> Self
    where
        T: Borrow<Self>,
    {
        let (pred_indexes_vec, pred_vec, target_vec) = iter
            .into_iter()
            .map(|(matchings, mini_batch_size)| {
                let Self {
                    pred_indexes,
                    pred,
                    target,
                } = matchings.borrow().shallow_clone();
                ((pred_indexes, mini_batch_size), pred, target)
            })
            .unzip_n_vec();

        Self {
            pred_indexes: FlatIndexTensor::cat_mini_batches(pred_indexes_vec),
            pred: ObjectDetectionTensor::cat(pred_vec),
            target: LabelTensor::cat(target_vec),
        }
    }
}
