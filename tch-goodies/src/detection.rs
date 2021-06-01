use crate::{
    bbox::{CyCxHW, TLBR},
    common::*,
    compound_tensor::{CyCxHWTensorUnchecked, DetectionTensor, DetectionTensorUnchecked},
    size::{GridSize, RatioSize, Size},
    unit::{GridUnit, PixelUnit, RatioUnit, Unit},
};

pub use dense_detection_tensor::*;
pub use flat_index_tensor::*;
pub use instance_index_tensor::*;
pub use label::*;
pub use merged_dense_detection::*;

mod merged_dense_detection {
    use super::*;

    #[derive(Debug, TensorLike)]
    pub struct MergedDenseDetection {
        inner: MergedDenseDetectionUnchecked,
    }

    impl MergedDenseDetection {
        pub fn from_detection_tensors(
            tensors: &[impl Borrow<DenseDetectionTensor>],
        ) -> Result<Self> {
            // ensure consistent sizes
            let (batch_size_set, num_classes_set): (HashSet<i64>, HashSet<usize>) = tensors
                .iter()
                .map(|tensor| {
                    let tensor = tensor.borrow();
                    (tensor.batch_size() as i64, tensor.num_classes())
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
                        obj,
                        class,
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

            let output = Self {
                inner: MergedDenseDetectionUnchecked {
                    cy,
                    cx,
                    h,
                    w,
                    obj,
                    class,
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
                        && feature_map.obj == detection.obj
                        && feature_map.class == detection.class
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
            let (_batch_size, num_classes, _instances) = self.class.size3().unwrap();
            num_classes
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
                                obj,
                                class,
                                ..
                            },
                    } = output.shallow_clone();

                    (batch_size, num_classes, info, cy, cx, h, w, obj, class)
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
            let obj = Tensor::cat(&obj_vec, 0);
            let class = Tensor::cat(&class_vec, 0);

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
                obj.size3()? == (batch_size, 1, flat_index_size),
                "invalid objectness shape"
            );
            ensure!(
                class.size3()? == (batch_size, num_classes as i64, flat_index_size),
                "invalid classification shape"
            );

            Ok(Self {
                inner: MergedDenseDetectionUnchecked {
                    cy,
                    cx,
                    h,
                    w,
                    obj,
                    class,
                    info,
                },
            })
        }

        pub fn index_by_flats(&self, flat_indexes: &FlatIndexTensor) -> DetectionTensor {
            let Self {
                inner:
                    MergedDenseDetectionUnchecked {
                        cy,
                        cx,
                        h,
                        w,
                        class,
                        obj,
                        ..
                    },
            } = self;
            let FlatIndexTensor { batches, flats } = flat_indexes;

            DetectionTensorUnchecked {
                cycxhw: CyCxHWTensorUnchecked {
                    cy: cy.index(&[Some(batches), None, Some(flats)]),
                    cx: cx.index(&[Some(batches), None, Some(flats)]),
                    h: h.index(&[Some(batches), None, Some(flats)]),
                    w: w.index(&[Some(batches), None, Some(flats)]),
                },
                obj: obj.index(&[Some(batches), None, Some(flats)]),
                class: class.index(&[Some(batches), None, Some(flats)]),
            }
            .try_into()
            .unwrap()
        }

        pub fn index_by_instances(
            &self,
            instance_indexes: &InstanceIndexTensor,
        ) -> DetectionTensor {
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
            let (batches, layers, anchors, grid_rows, grid_cols) =
                tuples?.into_iter().unzip_n_vec();

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
        pub obj: Tensor,
        /// Tensor of confidence scores per class of bboxes with shape `[batch, num_classes, flat]`.
        pub class: Tensor,
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
        pub anchors: Vec<RatioSize<R64>>,
        #[tensor_like(clone)]
        pub flat_index_range: Range<i64>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, TensorLike)]
    pub struct FlatIndex {
        pub batch_index: i64,
        pub flat_index: i64,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, TensorLike)]
    pub struct InstanceIndex {
        pub batch_index: i64,
        pub layer_index: i64,
        pub anchor_index: i64,
        pub grid_row: i64,
        pub grid_col: i64,
    }
}

mod dense_detection_tensor {
    use super::*;

    #[derive(Debug, TensorLike)]
    pub struct DenseDetectionTensorList {
        pub(super) inner: DenseDetectionTensorListUnchecked,
    }

    impl DenseDetectionTensorList {
        pub fn from_detection_tensors(
            tensors: &[impl Borrow<DenseDetectionTensor>],
        ) -> Result<Self> {
            let (tensors, batch_size_set): (Vec<_>, HashSet<_>) = tensors
                .iter()
                .map(|tensor| {
                    let tensor = tensor.borrow().shallow_clone();
                    let batch_size = tensor.batch_size();
                    (tensor, batch_size)
                })
                .unzip();

            ensure!(!tensors.is_empty());
            ensure!(batch_size_set.len() == 1);

            Ok(Self {
                inner: DenseDetectionTensorListUnchecked { tensors },
            })
        }

        pub fn batch_size(&self) -> usize {
            self.tensors[0].batch_size()
        }

        pub fn cat_batch(tensors: impl IntoIterator<Item = impl Borrow<Self>>) -> Result<Self> {
            Self::cat(tensors, 0)
        }

        pub fn cat_height(tensors: impl IntoIterator<Item = impl Borrow<Self>>) -> Result<Self> {
            Self::cat(tensors, 3)
        }

        pub fn cat_width(tensors: impl IntoIterator<Item = impl Borrow<Self>>) -> Result<Self> {
            Self::cat(tensors, 4)
        }

        fn cat(lists: impl IntoIterator<Item = impl Borrow<Self>>, index: i64) -> Result<Self> {
            // list index -> layer index -> tensor
            let tensors_vec: Vec<Vec<_>> = lists
                .into_iter()
                .map(|list| list.borrow().tensors.shallow_clone())
                .collect();

            // layer index -> list index -> tensor
            let tensors_vec = tensors_vec.transpose().unwrap();

            // concatenate each layer of tensors
            let tensors: Vec<_> = tensors_vec
                .into_iter()
                .map(|layer| DenseDetectionTensor::cat(layer, index))
                .try_collect()?;

            Ok(Self {
                inner: DenseDetectionTensorListUnchecked { tensors },
            })
        }
    }

    #[derive(Debug, TensorLike)]
    pub struct DenseDetectionTensorListUnchecked {
        pub tensors: Vec<DenseDetectionTensor>,
    }

    impl Borrow<DenseDetectionTensorListUnchecked> for DenseDetectionTensorList {
        fn borrow(&self) -> &DenseDetectionTensorListUnchecked {
            &self.inner
        }
    }

    impl Deref for DenseDetectionTensorList {
        type Target = DenseDetectionTensorListUnchecked;

        fn deref(&self) -> &Self::Target {
            &self.inner
        }
    }

    impl From<DenseDetectionTensorList> for DenseDetectionTensorListUnchecked {
        fn from(from: DenseDetectionTensorList) -> Self {
            from.inner
        }
    }

    impl TryFrom<DenseDetectionTensorListUnchecked> for DenseDetectionTensorList {
        type Error = Error;

        fn try_from(from: DenseDetectionTensorListUnchecked) -> Result<Self, Self::Error> {
            let DenseDetectionTensorListUnchecked { tensors } = &from;

            let (batch_size_set, num_classes_set): (HashSet<_>, HashSet<_>) = tensors
                .iter()
                .map(|tensor| (tensor.batch_size(), tensor.num_classes()))
                .unzip();

            ensure!(num_classes_set.len() == 1);
            ensure!(batch_size_set.len() == 1);

            Ok(Self { inner: from })
        }
    }

    /// Represents the output feature map of a layer.
    ///
    /// Every belonging tensor has shape `[batch, entry, anchor, height, width]`.
    #[derive(Debug, TensorLike, Getters)]
    pub struct DenseDetectionTensor {
        pub(super) inner: DenseDetectionTensorUnchecked,
    }

    impl DenseDetectionTensor {
        pub fn batch_size(&self) -> usize {
            let (batch_size, _, _, _, _) = self.cy.size5().unwrap();
            batch_size as usize
        }

        pub fn num_classes(&self) -> usize {
            let (_, num_classes, _, _, _) = self.class.size5().unwrap();
            num_classes as usize
        }

        pub fn num_anchors(&self) -> usize {
            let (_, _, num_anchors, _, _) = self.cy.size5().unwrap();
            num_anchors as usize
        }

        pub fn height(&self) -> usize {
            let (_, _, _, height, _) = self.cy.size5().unwrap();
            height as usize
        }

        pub fn width(&self) -> usize {
            let (_, _, _, _, width) = self.cy.size5().unwrap();
            width as usize
        }

        pub fn cat_batch(tensors: impl IntoIterator<Item = impl Borrow<Self>>) -> Result<Self> {
            Self::cat(tensors, 0)
        }

        pub fn cat_height(tensors: impl IntoIterator<Item = impl Borrow<Self>>) -> Result<Self> {
            Self::cat(tensors, 3)
        }

        pub fn cat_width(tensors: impl IntoIterator<Item = impl Borrow<Self>>) -> Result<Self> {
            Self::cat(tensors, 4)
        }

        fn cat(tensors: impl IntoIterator<Item = impl Borrow<Self>>, index: i64) -> Result<Self> {
            let (
                batch_size_set,
                num_classes_set,
                anchors_set,
                cy_vec,
                cx_vec,
                h_vec,
                w_vec,
                obj_vec,
                class_vec,
            ): (
                HashSet<_>,
                HashSet<_>,
                HashSet<_>,
                Vec<_>,
                Vec<_>,
                Vec<_>,
                Vec<_>,
                Vec<_>,
                Vec<_>,
            ) = tensors
                .into_iter()
                .map(|tensor| {
                    let tensor = tensor.borrow().shallow_clone();
                    let batch_size = tensor.batch_size();
                    let num_classes = tensor.num_classes();
                    let DenseDetectionTensorUnchecked {
                        cy,
                        cx,
                        h,
                        w,
                        obj,
                        class,
                        anchors,
                    } = tensor.into();
                    (batch_size, num_classes, anchors, cy, cx, h, w, obj, class)
                })
                .unzip_n();

            ensure!(batch_size_set.len() == 1);
            ensure!(num_classes_set.len() == 1);
            ensure!(anchors_set.len() == 1);

            let anchors = anchors_set.into_iter().next().unwrap();

            let cy = Tensor::cat(&cy_vec, index);
            let cx = Tensor::cat(&cx_vec, index);
            let h = Tensor::cat(&h_vec, index);
            let w = Tensor::cat(&w_vec, index);
            let obj = Tensor::cat(&obj_vec, index);
            let class = Tensor::cat(&class_vec, index);

            Ok(Self {
                inner: DenseDetectionTensorUnchecked {
                    cy,
                    cx,
                    h,
                    w,
                    obj,
                    class,
                    anchors: anchors.to_owned(),
                },
            })
        }
    }

    #[derive(Debug, TensorLike)]
    pub struct DenseDetectionTensorUnchecked {
        /// The bounding box center y position in ratio unit. It has 1 entry.
        pub cy: Tensor,
        /// The bounding box center x position in ratio unit. It has 1 entry.
        pub cx: Tensor,
        /// The bounding box height in ratio unit. It has 1 entry.
        pub h: Tensor,
        /// The bounding box width in ratio unit. It has 1 entry.
        pub w: Tensor,
        /// The likelihood score an object in the position. It has 1 entry.
        pub obj: Tensor,
        /// The scores the object is of that class. It number of entries is the number of classes.
        pub class: Tensor,
        #[tensor_like(clone)]
        pub anchors: Vec<RatioSize<R64>>,
    }

    impl Deref for DenseDetectionTensor {
        type Target = DenseDetectionTensorUnchecked;

        fn deref(&self) -> &Self::Target {
            &self.inner
        }
    }

    impl Borrow<DenseDetectionTensorUnchecked> for DenseDetectionTensor {
        fn borrow(&self) -> &DenseDetectionTensorUnchecked {
            &self.inner
        }
    }

    impl From<DenseDetectionTensor> for DenseDetectionTensorUnchecked {
        fn from(from: DenseDetectionTensor) -> Self {
            from.inner
        }
    }

    impl TryFrom<DenseDetectionTensorUnchecked> for DenseDetectionTensor {
        type Error = Error;

        fn try_from(from: DenseDetectionTensorUnchecked) -> Result<Self, Self::Error> {
            let DenseDetectionTensorUnchecked {
                cy,
                cx,
                h,
                w,
                obj,
                class,
                anchors,
            } = &from;

            let (batch_size, _num_classes, num_anchors, height, width) = class.size5()?;
            ensure!(cy.size5()? == (batch_size, 1, num_anchors, height, width),);
            ensure!(cx.size5()? == (batch_size, 1, num_anchors, height, width),);
            ensure!(h.size5()? == (batch_size, 1, num_anchors, height, width),);
            ensure!(w.size5()? == (batch_size, 1, num_anchors, height, width),);
            ensure!(obj.size5()? == (batch_size, 1, num_anchors, height, width),);
            ensure!(anchors.len() == num_anchors as usize);

            Ok(Self { inner: from })
        }
    }
}

mod label {
    use super::*;

    /// Generic bounding box with an extra class ID.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct Label<T, U>
    where
        T: Float,
        U: Unit,
    {
        pub cycxhw: CyCxHW<T, U>,
        pub class: usize,
    }

    pub type RatioLabel = Label<R64, RatioUnit>;
    pub type PixelLabel = Label<R64, PixelUnit>;
    pub type GridLabel = Label<R64, GridUnit>;

    impl<T, U> Label<T, U>
    where
        T: Float,
        U: Unit,
    {
        pub fn cy(&self) -> T {
            self.cycxhw.cy()
        }

        pub fn cx(&self) -> T {
            self.cycxhw.cx()
        }

        pub fn h(&self) -> T {
            self.cycxhw.h()
        }

        pub fn w(&self) -> T {
            self.cycxhw.w()
        }

        pub fn tlbr(&self) -> TLBR<T, U> {
            (&self.cycxhw).into()
        }

        pub fn size(&self) -> Size<T, U> {
            self.cycxhw.size()
        }

        pub fn area(&self) -> T {
            self.cycxhw.area()
        }

        /// Compute intersection area in TLBR format.
        pub fn intersect_with(&self, other: &CyCxHW<T, U>) -> Option<Self> {
            let intersection: CyCxHW<_, _> =
                (&TLBR::from(&self.cycxhw).intersect_with(&other.into())?).into();

            Some(Self {
                cycxhw: intersection,
                class: self.class,
            })
        }

        pub fn scale_size(&self, scale: T) -> Result<Self> {
            let Self { ref cycxhw, class } = *self;
            Ok(Self {
                cycxhw: cycxhw.scale_size(scale)?,
                class,
            })
        }
    }

    impl<T, U> AsRef<CyCxHW<T, U>> for Label<T, U>
    where
        T: Float,
        U: Unit,
    {
        fn as_ref(&self) -> &CyCxHW<T, U> {
            &self.cycxhw
        }
    }
}

mod flat_index_tensor {
    use super::*;

    #[derive(Debug, TensorLike)]
    pub struct FlatIndexTensor {
        pub batches: Tensor,
        pub flats: Tensor,
    }

    impl FlatIndexTensor {
        pub fn num_samples(&self) -> i64 {
            self.batches.size1().unwrap()
        }

        pub fn cat_mini_batches<T>(indexes: impl IntoIterator<Item = (T, i64)>) -> Self
        where
            T: Borrow<Self>,
        {
            let (batches_vec, flats_vec) = indexes
                .into_iter()
                .scan(0i64, |batch_base, (flat_indexes, mini_batch_size)| {
                    let Self { batches, flats } = flat_indexes.borrow().shallow_clone();

                    let new_batches = batches + *batch_base;
                    *batch_base += mini_batch_size;

                    Some((new_batches, flats))
                })
                .unzip_n_vec();

            Self {
                batches: Tensor::cat(&batches_vec, 0),
                flats: Tensor::cat(&flats_vec, 0),
            }
        }
    }

    impl<'a> From<&'a FlatIndexTensor> for Vec<FlatIndex> {
        fn from(from: &'a FlatIndexTensor) -> Self {
            let FlatIndexTensor { batches, flats } = from;

            izip!(Vec::<i64>::from(batches), Vec::<i64>::from(flats))
                .map(|(batch_index, flat_index)| FlatIndex {
                    batch_index,
                    flat_index,
                })
                .collect()
        }
    }

    impl From<FlatIndexTensor> for Vec<FlatIndex> {
        fn from(from: FlatIndexTensor) -> Self {
            (&from).into()
        }
    }

    impl FromIterator<FlatIndex> for FlatIndexTensor {
        fn from_iter<T: IntoIterator<Item = FlatIndex>>(iter: T) -> Self {
            let (batches, flats) = iter
                .into_iter()
                .map(|index| {
                    let FlatIndex {
                        batch_index,
                        flat_index,
                    } = index;
                    (batch_index, flat_index)
                })
                .unzip_n_vec();
            Self {
                batches: Tensor::of_slice(&batches),
                flats: Tensor::of_slice(&flats),
            }
        }
    }

    impl<'a> FromIterator<&'a FlatIndex> for FlatIndexTensor {
        fn from_iter<T: IntoIterator<Item = &'a FlatIndex>>(iter: T) -> Self {
            let (batches, flats) = iter
                .into_iter()
                .map(|index| {
                    let FlatIndex {
                        batch_index,
                        flat_index,
                    } = *index;
                    (batch_index, flat_index)
                })
                .unzip_n_vec();
            Self {
                batches: Tensor::of_slice(&batches),
                flats: Tensor::of_slice(&flats),
            }
        }
    }
}

mod instance_index_tensor {
    use super::*;

    #[derive(Debug, TensorLike)]
    pub struct InstanceIndexTensor {
        pub batches: Tensor,
        pub layers: Tensor,
        pub anchors: Tensor,
        pub grid_rows: Tensor,
        pub grid_cols: Tensor,
    }

    impl FromIterator<InstanceIndex> for InstanceIndexTensor {
        fn from_iter<T: IntoIterator<Item = InstanceIndex>>(iter: T) -> Self {
            let (batches, layers, anchors, grid_rows, grid_cols) = iter
                .into_iter()
                .map(|index| {
                    let InstanceIndex {
                        batch_index,
                        layer_index,
                        anchor_index,
                        grid_row,
                        grid_col,
                    } = index;
                    (batch_index, layer_index, anchor_index, grid_row, grid_col)
                })
                .unzip_n_vec();
            Self {
                batches: Tensor::of_slice(&batches),
                layers: Tensor::of_slice(&layers),
                anchors: Tensor::of_slice(&anchors),
                grid_rows: Tensor::of_slice(&grid_rows),
                grid_cols: Tensor::of_slice(&grid_cols),
            }
        }
    }

    impl<'a> FromIterator<&'a InstanceIndex> for InstanceIndexTensor {
        fn from_iter<T: IntoIterator<Item = &'a InstanceIndex>>(iter: T) -> Self {
            let (batches, layers, anchors, grid_rows, grid_cols) = iter
                .into_iter()
                .map(|index| {
                    let InstanceIndex {
                        batch_index,
                        layer_index,
                        anchor_index,
                        grid_row,
                        grid_col,
                    } = *index;
                    (batch_index, layer_index, anchor_index, grid_row, grid_col)
                })
                .unzip_n_vec();
            Self {
                batches: Tensor::of_slice(&batches),
                layers: Tensor::of_slice(&layers),
                anchors: Tensor::of_slice(&anchors),
                grid_rows: Tensor::of_slice(&grid_rows),
                grid_cols: Tensor::of_slice(&grid_cols),
            }
        }
    }
}

mod convert {
    use super::*;

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
}
