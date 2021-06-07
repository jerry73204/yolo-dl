use super::{
    DenseDetectionTensor, DenseDetectionTensorList, DenseDetectionTensorListUnchecked,
    DenseDetectionTensorUnchecked, DetectionInfo, MergedDenseDetection,
};
use crate::common::*;

impl TryFrom<DenseDetectionTensorList> for MergedDenseDetection {
    type Error = Error;

    fn try_from(from: DenseDetectionTensorList) -> Result<Self, Self::Error> {
        Self::from_detection_tensors(&from.tensors)
    }
}

impl From<MergedDenseDetection> for DenseDetectionTensorList {
    fn from(from: MergedDenseDetection) -> DenseDetectionTensorList {
        let batch_size = from.batch_size();
        let num_classes = from.num_classes();

        let tensors = from
            .info
            .iter()
            .enumerate()
            .map(|(_layer_index, meta)| {
                let DetectionInfo {
                    ref feature_size,
                    ref anchors,
                    ref flat_index_range,
                    ..
                } = *meta;
                let num_anchors = anchors.len() as i64;
                let [feature_h, feature_w] = feature_size.hw_params();

                let cy_map = from.cy.i((.., .., flat_index_range.clone())).view([
                    batch_size,
                    1,
                    num_anchors,
                    feature_h,
                    feature_w,
                ]);
                let cx_map = from.cx.i((.., .., flat_index_range.clone())).view([
                    batch_size,
                    1,
                    num_anchors,
                    feature_h,
                    feature_w,
                ]);
                let h_map = from.h.i((.., .., flat_index_range.clone())).view([
                    batch_size,
                    1,
                    num_anchors,
                    feature_h,
                    feature_w,
                ]);
                let w_map = from.w.i((.., .., flat_index_range.clone())).view([
                    batch_size,
                    1,
                    num_anchors,
                    feature_h,
                    feature_w,
                ]);
                let obj_map = from.obj.i((.., .., flat_index_range.clone())).view([
                    batch_size,
                    1,
                    num_anchors,
                    feature_h,
                    feature_w,
                ]);
                let class_map = from.class.i((.., .., flat_index_range.clone())).view([
                    batch_size,
                    num_classes as i64,
                    num_anchors,
                    feature_h,
                    feature_w,
                ]);

                DenseDetectionTensor {
                    inner: DenseDetectionTensorUnchecked {
                        cy: cy_map,
                        cx: cx_map,
                        h: h_map,
                        w: w_map,
                        obj: obj_map,
                        class: class_map,
                        anchors: anchors.clone(),
                    },
                }
            })
            .collect_vec();

        DenseDetectionTensorList {
            inner: DenseDetectionTensorListUnchecked { tensors },
        }
    }
}
