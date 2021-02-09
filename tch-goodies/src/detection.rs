use crate::{
    common::*,
    size::{GridSize, PixelSize},
};

#[derive(Debug, TensorLike)]
pub struct DenseDetectionInit {
    pub anchors: Vec<GridSize<f64>>,
    pub num_classes: usize,
    pub bbox_cy: Tensor,
    pub bbox_cx: Tensor,
    pub bbox_h: Tensor,
    pub bbox_w: Tensor,
    pub objectness: Tensor,
    pub classification: Tensor,
}

#[derive(Debug, Getters, TensorLike)]
pub struct DenseDetection {
    #[getset(get_copy = "pub")]
    batch_size: i64,
    #[getset(get_copy = "pub")]
    num_classes: usize,
    #[getset(get_copy = "pub")]
    #[tensor_like(copy)]
    device: Device,
    #[getset(get = "pub")]
    anchors: Vec<GridSize<f64>>,
    #[getset(get = "pub")]
    feature_size: GridSize<i64>,
    #[getset(get = "pub")]
    bbox_cy: Tensor,
    #[getset(get = "pub")]
    bbox_cx: Tensor,
    #[getset(get = "pub")]
    bbox_h: Tensor,
    #[getset(get = "pub")]
    bbox_w: Tensor,
    #[getset(get = "pub")]
    objectness: Tensor,
    #[getset(get = "pub")]
    classification: Tensor,
}

impl TryFrom<DenseDetectionInit> for DenseDetection {
    type Error = Error;

    fn try_from(from: DenseDetectionInit) -> Result<Self, Self::Error> {
        let DenseDetectionInit {
            anchors,
            num_classes,
            bbox_cy,
            bbox_cx,
            bbox_h,
            bbox_w,
            objectness,
            classification,
        } = from;

        let num_anchors = anchors.len() as i64;

        // ensure all tensors are on the same device
        let device = {
            let mut iter = vec![
                bbox_cy.device(),
                bbox_cx.device(),
                bbox_h.device(),
                bbox_w.device(),
                objectness.device(),
                classification.device(),
            ]
            .into_iter();
            let first = iter.next().unwrap();
            ensure!(
                iter.all(|dev| first == dev),
                "all tensors must be on the same device"
            );
            first
        };

        // ensure all tensors have Float kind
        {
            let kinds = vec![
                bbox_cy.kind(),
                bbox_cx.kind(),
                bbox_h.kind(),
                bbox_w.kind(),
                objectness.kind(),
                classification.kind(),
            ];
            ensure!(
                kinds.into_iter().all(|kind| matches!(kind, Kind::Float)),
                "all tensors must have float kind"
            );
        }

        // ensure every tensor has shape (batch_size x num_entries x num_anchors x height x width)
        let (batch_size, feature_size) = {
            let (batch_size, _entries, _anchors, height, width) = bbox_cy.size5()?;
            ensure!(
                bbox_cy.size5()? == (batch_size, 1, num_anchors, height, width),
                "bbox_cy has invalid shape"
            );
            ensure!(
                bbox_cx.size5()? == (batch_size, 1, num_anchors, height, width),
                "bbox_cx has invalid shape"
            );
            ensure!(
                bbox_h.size5()? == (batch_size, 1, num_anchors, height, width),
                "bbox_h has invalid shape"
            );
            ensure!(
                bbox_w.size5()? == (batch_size, 1, num_anchors, height, width),
                "bbox_w has invalid shape"
            );
            ensure!(
                objectness.size5()? == (batch_size, 1, num_anchors, height, width),
                "objectness has invalid shape"
            );
            ensure!(
                classification.size5()?
                    == (batch_size, num_classes as i64, num_anchors, height, width),
                "classification has invalid shape"
            );

            (batch_size, GridSize::new(height, width))
        };

        Ok(Self {
            batch_size,
            feature_size,
            anchors,
            num_classes,
            device,
            bbox_cy,
            bbox_cx,
            bbox_h,
            bbox_w,
            objectness,
            classification,
        })
    }
}

#[derive(Debug, Getters, TensorLike)]
pub struct MultiDenseDetection {
    #[getset(get = "pub")]
    image_size: PixelSize<i64>,
    #[getset(get_copy = "pub")]
    batch_size: i64,
    #[getset(get_copy = "pub")]
    num_classes: usize,
    #[getset(get_copy = "pub")]
    #[tensor_like(copy)]
    #[getset(get = "pub")]
    device: Device,
    #[getset(get = "pub")]
    layer_meta: Vec<LayerMeta>,
    // below tensors are indexed by (batch x entry x flat_index), where
    // flat_index is ( \sum_i anchor_i x height_i x width_i )
    #[getset(get = "pub")]
    bbox_cy: Tensor,
    #[getset(get = "pub")]
    bbox_cx: Tensor,
    #[getset(get = "pub")]
    bbox_h: Tensor,
    #[getset(get = "pub")]
    bbox_w: Tensor,
    #[getset(get = "pub")]
    objectness: Tensor,
    #[getset(get = "pub")]
    classification: Tensor,
}

impl MultiDenseDetection {
    pub fn new(
        image_height: usize,
        image_width: usize,
        detections: impl IntoIterator<Item = DenseDetection>,
    ) -> Result<Self> {
        // unzip iterator
        let (
            batch_size_set,
            num_classes_set,
            meta_vec,
            device_vec,
            bbox_cy_vec,
            bbox_cx_vec,
            bbox_h_vec,
            bbox_w_vec,
            objectness_vec,
            classification_vec,
        ): (
            HashSet<_>,
            HashSet<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
        ) = detections
            .into_iter()
            .scan(0, |flat_index, detection| {
                let DenseDetection {
                    batch_size,
                    feature_size,
                    anchors,
                    num_classes,
                    device,
                    bbox_cy,
                    bbox_cx,
                    bbox_h,
                    bbox_w,
                    objectness,
                    classification,
                } = detection;

                let num_anchors = anchors.len() as i64;
                let GridSize {
                    h: feature_h,
                    w: feature_w,
                    ..
                } = feature_size;

                // compute flat index
                let flat_index_range = {
                    let num_flat_indexes = num_anchors * feature_h * feature_w;
                    let begin = *flat_index;
                    let end = begin + num_flat_indexes;

                    // update state
                    *flat_index = end;

                    begin..end
                };

                let grid_size = {
                    let grid_h = image_height as f64 / feature_h as f64;
                    let grid_w = image_width as f64 / feature_w as f64;
                    PixelSize::new(grid_h, grid_w)
                };

                let meta = LayerMeta {
                    feature_size,
                    grid_size,
                    anchors,
                    flat_index_range,
                };

                Some((
                    batch_size,
                    num_classes,
                    meta,
                    device,
                    bbox_cy.view([batch_size, 1, -1]),
                    bbox_cx.view([batch_size, 1, -1]),
                    bbox_h.view([batch_size, 1, -1]),
                    bbox_w.view([batch_size, 1, -1]),
                    objectness.view([batch_size, 1, -1]),
                    classification.view([batch_size, num_classes as i64, -1]),
                ))
            })
            .unzip_n();

        // ensure non-empty
        ensure!(
            !batch_size_set.is_empty(),
            "at least one dense detection must be given"
        );

        // ensure batch_sizes are equal
        let batch_size = {
            ensure!(
                batch_size_set.len() == 1,
                "batch sizes of every detection must be equal"
            );
            batch_size_set.into_iter().next().unwrap()
        };

        // ensure num_classes are equal
        let num_classes = {
            ensure!(
                num_classes_set.len() == 1,
                "number of classes of every detection must be equal"
            );
            num_classes_set.into_iter().next().unwrap()
        };

        // ensure devices must be equal
        let device = {
            let mut iter = device_vec.into_iter();
            let first = iter.next().unwrap();
            ensure!(
                iter.all(|dev| first == dev),
                "device of every detection must be equal"
            );
            first
        };

        // image size
        let image_size = PixelSize::new(image_height as i64, image_width as i64);

        // concatenate tensors
        let bbox_cy = Tensor::cat(&bbox_cy_vec, 2);
        let bbox_cx = Tensor::cat(&bbox_cx_vec, 2);
        let bbox_h = Tensor::cat(&bbox_h_vec, 2);
        let bbox_w = Tensor::cat(&bbox_w_vec, 2);
        let objectness = Tensor::cat(&objectness_vec, 2);
        let classification = Tensor::cat(&classification_vec, 2);

        Ok(Self {
            image_size,
            batch_size,
            num_classes,
            device,
            layer_meta: meta_vec,
            bbox_cy,
            bbox_cx,
            bbox_h,
            bbox_w,
            objectness,
            classification,
        })
    }

    pub fn to_flat_index(&self, instance_index: &InstanceIndex) -> i64 {
        let InstanceIndex {
            layer_index,
            anchor_index,
            grid_row,
            grid_col,
            ..
        } = *instance_index;

        let LayerMeta {
            ref flat_index_range,
            feature_size: GridSize { h, w, .. },
            ..
        } = self.layer_meta[layer_index];

        flat_index_range.start + grid_col + w * (grid_row + h * anchor_index)
    }
}

#[derive(Debug, TensorLike)]
pub struct LayerMeta {
    /// feature map size in grid units
    pub feature_size: GridSize<i64>,
    /// per grid size in pixel units
    pub grid_size: PixelSize<f64>,
    /// Anchros (height, width) in grid units
    pub anchors: Vec<GridSize<f64>>,
    #[tensor_like(clone)]
    pub flat_index_range: Range<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, TensorLike)]
pub struct InstanceIndex {
    pub batch_index: usize,
    pub layer_index: usize,
    pub anchor_index: i64,
    pub grid_row: i64,
    pub grid_col: i64,
}
