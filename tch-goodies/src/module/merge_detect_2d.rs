use super::*;
use crate::{
    common::*,
    detection::{DetectionInfo, MergedDenseDetection},
    size::GridSize,
};

#[derive(Debug)]
pub struct MergeDetect2D {
    _private: [u8; 0],
}

impl MergeDetect2D {
    pub fn new() -> Self {
        Self { _private: [] }
    }

    pub fn forward(&mut self, detections: &[&Detect2DOutput]) -> Result<MergeDetect2DOutput> {
        // ensure consistent sizes
        let (batch_size_set, num_classes_set): (HashSet<i64>, HashSet<usize>) = detections
            .iter()
            .cloned()
            .map(|detection| (detection.batch_size() as i64, detection.num_classes()))
            .unzip_n();

        ensure!(batch_size_set.len() == 1, "TODO");
        ensure!(num_classes_set.len() == 1, "TODO");
        let num_classes = num_classes_set.into_iter().next().unwrap();

        // merge detections
        let (cy_vec, cx_vec, h_vec, w_vec, obj_vec, class_vec, info) = detections
            .iter()
            .cloned()
            .scan(0, |base_flat_index, detection| {
                let Detect2DOutput {
                    ref anchors,
                    ref cy,
                    ref cx,
                    ref h,
                    ref w,
                    ref obj,
                    ref class,
                    ..
                } = *detection;
                let batch_size = detection.batch_size() as i64;
                let feature_h = detection.height() as i64;
                let feature_w = detection.width() as i64;
                let num_anchors = detection.num_anchors();

                // flatten tensors
                let cy_flat = cy.view([batch_size, 1, -1]);
                let cx_flat = cx.view([batch_size, 1, -1]);
                let h_flat = h.view([batch_size, 1, -1]);
                let w_flat = w.view([batch_size, 1, -1]);
                let obj_flat = obj.view([batch_size, 1, -1]);
                let class_flat = class.view([batch_size, num_classes as i64, -1]);

                // save feature anchors and shapes
                let info = {
                    let begin_flat_index = *base_flat_index;
                    *base_flat_index += num_anchors as i64 * feature_h * feature_w;
                    let end_flat_index = *base_flat_index;

                    // compute base flat index

                    DetectionInfo {
                        feature_size: GridSize::new(feature_h, feature_w).unwrap(),
                        anchors: anchors.to_owned(),
                        flat_index_range: begin_flat_index..end_flat_index,
                    }
                };

                Some((cy_flat, cx_flat, h_flat, w_flat, obj_flat, class_flat, info))
            })
            .unzip_n_vec();

        let cy = Tensor::cat(&cy_vec, 2);
        let cx = Tensor::cat(&cx_vec, 2);
        let h = Tensor::cat(&h_vec, 2);
        let w = Tensor::cat(&w_vec, 2);
        let obj = Tensor::cat(&obj_vec, 2);
        let class = Tensor::cat(&class_vec, 2);

        let output = MergeDetect2DOutput {
            num_classes,
            cy,
            cx,
            h,
            w,
            obj,
            class,
            info,
        };

        debug_assert!({
            let feature_maps = output.feature_maps();
            izip!(feature_maps, detections).all(|(feature_map, detection)| {
                feature_map.cy == detection.cy
                    && feature_map.cx == detection.cx
                    && feature_map.h == detection.h
                    && feature_map.w == detection.w
                    && feature_map.obj == detection.obj
                    && feature_map.class == detection.class
            })
        });

        Ok(output)
    }
}

pub type MergeDetect2DOutput = MergedDenseDetection;
