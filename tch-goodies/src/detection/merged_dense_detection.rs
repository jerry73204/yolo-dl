use super::{
    DenseDetectionTensor, DenseDetectionTensorList, DenseDetectionTensorUnchecked, FlatIndex,
    FlatIndexTensor, InstanceIndex, InstanceIndexTensor, ObjectDetectionTensor,
    ObjectDetectionTensorUnchecked,
};
use crate::{common::*, compound_tensor::CyCxHWTensorUnchecked, size::GridSize};

#[derive(Debug, TensorLike)]
pub struct MergedDenseDetection {
    inner: MergedDenseDetectionUnchecked,
}

impl MergedDenseDetection {
    pub fn from_detection_tensors(
        tensors: impl IntoIterator<Item = impl Borrow<DenseDetectionTensor>>,
    ) -> Result<Self> {
        // ensure consistent sizes
        let (tensors, batch_size_set, num_classes_set): (Vec<_>, HashSet<i64>, HashSet<usize>) =
            tensors
                .into_iter()
                .map(|tensor| {
                    let borrow = tensor.borrow();
                    let batch_size = borrow.batch_size() as i64;
                    let num_classes = borrow.num_classes();
                    (tensor, batch_size, num_classes)
                })
                .unzip_n();

        ensure!(batch_size_set.len() == 1, "TODO");
        ensure!(num_classes_set.len() == 1, "TODO");
        let num_classes = num_classes_set.into_iter().next().unwrap();

        // merge detections
        let (cy_vec, cx_vec, h_vec, w_vec, obj_vec, class_vec, info) = tensors
            .iter()
            .scan(0, |base_flat_index, detection| {
                let detection = detection.borrow();

                let DenseDetectionTensorUnchecked {
                    anchors,
                    cy,
                    cx,
                    h,
                    w,
                    obj_logit,
                    class_logit,
                    ..
                } = &**detection;
                let batch_size = detection.batch_size() as i64;
                let feature_h = detection.height() as i64;
                let feature_w = detection.width() as i64;
                let num_anchors = detection.num_anchors();

                // flatten tensors
                let cy_flat = cy.view([batch_size, 1, -1]);
                let cx_flat = cx.view([batch_size, 1, -1]);
                let h_flat = h.view([batch_size, 1, -1]);
                let w_flat = w.view([batch_size, 1, -1]);
                let obj_flat = obj_logit.view([batch_size, 1, -1]);
                let class_flat = class_logit.view([batch_size, num_classes as i64, -1]);

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
        let obj_logit = Tensor::cat(&obj_vec, 2);
        let class_logit = Tensor::cat(&class_vec, 2);

        let output = Self {
            inner: MergedDenseDetectionUnchecked {
                cy,
                cx,
                h,
                w,
                obj_logit,
                class_logit,
                info,
            },
        };

        debug_assert!({
            let feature_maps = output.to_tensor_list();
            izip!(&feature_maps.tensors, tensors).all(|(feature_map, detection)| {
                let detection = detection.borrow();
                feature_map.cy == detection.cy
                    && feature_map.cx == detection.cx
                    && feature_map.h == detection.h
                    && feature_map.w == detection.w
                    && feature_map.obj_logit == detection.obj_logit
                    && feature_map.class_logit == detection.class_logit
            })
        });

        Ok(output)
    }

    /// Gets the device of belonging tensors.
    pub fn device(&self) -> Device {
        self.cy.device()
    }

    /// Gets the batch size of belonging tensors.
    pub fn batch_size(&self) -> i64 {
        let (batch_size, _entries, _instances) = self.cy.size3().unwrap();
        batch_size
    }

    pub fn num_instances(&self) -> i64 {
        let (_batch_size, _entries, instances) = self.cy.size3().unwrap();
        instances
    }

    pub fn num_classes(&self) -> i64 {
        let (_batch_size, num_classes, _instances) = self.class_logit.size3().unwrap();
        num_classes
    }

    /// Compute confidence, objectness score times classification score.
    pub fn confidence(&self) -> Tensor {
        self.obj_prob() * self.class_prob()
    }

    pub fn obj_prob(&self) -> Tensor {
        self.inner.obj_logit.sigmoid()
    }

    pub fn class_prob(&self) -> Tensor {
        self.inner.class_logit.sigmoid()
    }

    pub fn cat(outputs: impl IntoIterator<Item = impl Borrow<Self>>) -> Result<Self> {
        let (
            batch_size_set,
            num_classes_set,
            info_set,
            cy_vec,
            cx_vec,
            h_vec,
            w_vec,
            obj_vec,
            class_vec,
        ): (
            HashSet<i64>,
            HashSet<i64>,
            HashSet<Vec<DetectionInfo>>,
            Vec<Tensor>,
            Vec<Tensor>,
            Vec<Tensor>,
            Vec<Tensor>,
            Vec<Tensor>,
            Vec<Tensor>,
        ) = outputs
            .into_iter()
            .map(|output| {
                let output = output.borrow();
                let batch_size = output.batch_size();
                let num_classes = output.num_classes();
                let Self {
                    inner:
                        MergedDenseDetectionUnchecked {
                            info,
                            cy,
                            cx,
                            h,
                            w,
                            obj_logit,
                            class_logit,
                            ..
                        },
                } = output.shallow_clone();

                (
                    batch_size,
                    num_classes,
                    info,
                    cy,
                    cx,
                    h,
                    w,
                    obj_logit,
                    class_logit,
                )
            })
            .unzip_n();

        let num_outputs = cy_vec.len();
        let batch_size = {
            ensure!(batch_size_set.len() == 1, "batch_size must be equal");
            batch_size_set.into_iter().next().unwrap() * num_outputs as i64
        };
        let num_classes = {
            ensure!(num_classes_set.len() == 1, "num_classes must be equal");
            num_classes_set.into_iter().next().unwrap()
        };
        let info = {
            ensure!(info_set.len() == 1, "detection info must be equal");
            info_set.into_iter().next().unwrap()
        };
        let cy = Tensor::cat(&cy_vec, 0);
        let cx = Tensor::cat(&cx_vec, 0);
        let h = Tensor::cat(&h_vec, 0);
        let w = Tensor::cat(&w_vec, 0);
        let obj_logit = Tensor::cat(&obj_vec, 0);
        let class_logit = Tensor::cat(&class_vec, 0);

        let flat_index_size: i64 = info
            .iter()
            .map(|meta| {
                let DetectionInfo {
                    ref feature_size,
                    ref anchors,
                    ..
                } = *meta;
                let [h, w] = feature_size.hw_params();
                h * w * anchors.len() as i64
            })
            .sum();

        ensure!(
            cy.size3()? == (batch_size, 1, flat_index_size),
            "invalid cy shape"
        );
        ensure!(
            cx.size3()? == (batch_size, 1, flat_index_size),
            "invalid cx shape"
        );
        ensure!(
            h.size3()? == (batch_size, 1, flat_index_size),
            "invalid height shape"
        );
        ensure!(
            w.size3()? == (batch_size, 1, flat_index_size),
            "invalid width shape"
        );
        ensure!(
            obj_logit.size3()? == (batch_size, 1, flat_index_size),
            "invalid objectness shape"
        );
        ensure!(
            class_logit.size3()? == (batch_size, num_classes as i64, flat_index_size),
            "invalid classification shape"
        );

        Ok(Self {
            inner: MergedDenseDetectionUnchecked {
                cy,
                cx,
                h,
                w,
                obj_logit,
                class_logit,
                info,
            },
        })
    }

    pub fn index_by_flats(&self, flat_indexes: &FlatIndexTensor) -> ObjectDetectionTensor {
        let Self {
            inner:
                MergedDenseDetectionUnchecked {
                    cy,
                    cx,
                    h,
                    w,
                    class_logit,
                    obj_logit,
                    ..
                },
        } = self;
        let FlatIndexTensor { batches, flats } = flat_indexes;

        ObjectDetectionTensorUnchecked {
            cycxhw: CyCxHWTensorUnchecked {
                cy: cy.index(&[Some(batches), None, Some(flats)]),
                cx: cx.index(&[Some(batches), None, Some(flats)]),
                h: h.index(&[Some(batches), None, Some(flats)]),
                w: w.index(&[Some(batches), None, Some(flats)]),
            },
            obj_logit: obj_logit.index(&[Some(batches), None, Some(flats)]),
            class_logit: class_logit.index(&[Some(batches), None, Some(flats)]),
        }
        .try_into()
        .unwrap()
    }

    pub fn index_by_instances(
        &self,
        instance_indexes: &InstanceIndexTensor,
    ) -> ObjectDetectionTensor {
        let flat_indexes = self.instances_to_flats(instance_indexes).unwrap();
        self.index_by_flats(&flat_indexes)
    }

    pub fn flat_to_instance_index(&self, flat_index: &FlatIndex) -> Option<InstanceIndex> {
        let FlatIndex {
            batch_index,
            flat_index,
        } = *flat_index;
        let batch_size = self.batch_size();
        if batch_index as i64 >= batch_size || flat_index < 0 {
            return None;
        }

        let (
            layer_index,
            DetectionInfo {
                feature_size,
                anchors,
                flat_index_range,
                ..
            },
        ) = self
            .info
            .iter()
            .enumerate()
            .find(|(_layer_index, meta)| flat_index < meta.flat_index_range.end)?;

        let remainder = flat_index - flat_index_range.start;
        let grid_col = remainder % feature_size.w();
        let grid_row = remainder / feature_size.w() % feature_size.h();
        let anchor_index = remainder / feature_size.w() / feature_size.h();

        if anchor_index >= anchors.len() as i64 {
            return None;
        }

        Some(InstanceIndex {
            batch_index: batch_index as i64,
            layer_index: layer_index as i64,
            anchor_index,
            grid_row,
            grid_col,
        })
    }

    pub fn instance_to_flat_index(&self, instance_index: &InstanceIndex) -> Option<FlatIndex> {
        let InstanceIndex {
            batch_index,
            layer_index,
            anchor_index,
            grid_row,
            grid_col,
        } = *instance_index;

        let DetectionInfo {
            ref flat_index_range,
            ref feature_size,
            ..
        } = self.info.get(layer_index as usize)?;

        let flat_index = flat_index_range.start
            + grid_col
            + feature_size.w() * (grid_row + feature_size.h() * anchor_index);

        Some(FlatIndex {
            batch_index,
            flat_index,
        })
    }

    pub fn flats_to_instances(
        &self,
        flat_indexes: &FlatIndexTensor,
    ) -> Option<InstanceIndexTensor> {
        let FlatIndexTensor { batches, flats } = flat_indexes;

        let tuples: Option<Vec<_>> = izip!(Vec::<i64>::from(batches), Vec::<i64>::from(flats))
            .map(|(batch_index, flat_index)| {
                let InstanceIndex {
                    batch_index,
                    layer_index,
                    anchor_index,
                    grid_row,
                    grid_col,
                } = self.flat_to_instance_index(&FlatIndex {
                    batch_index,
                    flat_index,
                })?;

                Some((batch_index, layer_index, anchor_index, grid_row, grid_col))
            })
            .collect();
        let (batches, layers, anchors, grid_rows, grid_cols) = tuples?.into_iter().unzip_n_vec();

        Some(InstanceIndexTensor {
            batches: Tensor::of_slice(&batches),
            layers: Tensor::of_slice(&layers),
            anchors: Tensor::of_slice(&anchors),
            grid_rows: Tensor::of_slice(&grid_rows),
            grid_cols: Tensor::of_slice(&grid_cols),
        })
    }

    pub fn instances_to_flats(
        &self,
        instance_indexes: &InstanceIndexTensor,
    ) -> Option<FlatIndexTensor> {
        let InstanceIndexTensor {
            batches,
            layers,
            anchors,
            grid_rows,
            grid_cols,
        } = instance_indexes;

        let tuples: Option<Vec<_>> = izip!(
            Vec::<i64>::from(batches),
            Vec::<i64>::from(layers),
            Vec::<i64>::from(anchors),
            Vec::<i64>::from(grid_rows),
            Vec::<i64>::from(grid_cols),
        )
        .map(
            |(batch_index, layer_index, anchor_index, grid_row, grid_col)| {
                let FlatIndex {
                    batch_index,
                    flat_index,
                } = self.instance_to_flat_index(&InstanceIndex {
                    batch_index,
                    layer_index,
                    anchor_index,
                    grid_row,
                    grid_col,
                })?;
                Some((batch_index, flat_index))
            },
        )
        .collect();
        let (batches, flats) = tuples?.into_iter().unzip_n_vec();

        Some(FlatIndexTensor {
            batches: Tensor::of_slice(&batches),
            flats: Tensor::of_slice(&flats),
        })
    }

    pub fn to_tensor_list(&self) -> DenseDetectionTensorList {
        self.shallow_clone().into()
    }
}

#[derive(Debug, TensorLike)]
pub struct MergedDenseDetectionUnchecked {
    /// Tensor of bbox center y coordinates with shape `[batch, 1, flat]`.
    pub cy: Tensor,
    /// Tensor of bbox center x coordinates with shape `[batch, 1, flat]`.
    pub cx: Tensor,
    /// Tensor of bbox heights with shape `[batch, 1, flat]`.
    pub h: Tensor,
    /// Tensor of bbox widths with shape `[batch, 1, flat]`.
    pub w: Tensor,
    /// Tensor of bbox objectness score with shape `[batch, 1, flat]`.
    pub obj_logit: Tensor,
    /// Tensor of confidence scores per class of bboxes with shape `[batch, num_classes, flat]`.
    pub class_logit: Tensor,
    /// Saves the shape of exported feature maps.
    pub info: Vec<DetectionInfo>,
}

impl Deref for MergedDenseDetection {
    type Target = MergedDenseDetectionUnchecked;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Borrow<MergedDenseDetectionUnchecked> for MergedDenseDetection {
    fn borrow(&self) -> &MergedDenseDetectionUnchecked {
        &self.inner
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, TensorLike)]
pub struct DetectionInfo {
    /// feature map size in grid units
    #[tensor_like(clone)]
    pub feature_size: GridSize<i64>,
    /// Anchros (height, width) in grid units
    #[tensor_like(clone)]
    pub anchors: Vec<GridSize<R64>>,
    #[tensor_like(clone)]
    pub flat_index_range: Range<i64>,
}
