use super::*;
use crate::common::*;

pub use misc::*;
pub use yolo_model::*;
pub use yolo_output::*;

mod yolo_model {
    use super::*;

    #[derive(Debug)]
    pub struct YoloModel {
        pub(crate) layers: Vec<Layer>,
        pub(crate) detection_module: DetectModule,
    }

    impl YoloModel {
        pub fn forward_t(&mut self, xs: &Tensor, train: bool) -> YoloOutput {
            let (_batch_size, _channels, height, width) = xs.size4().unwrap();
            let image_size = PixelSize::new(height, width);
            let mut tmp_tensors: HashMap<usize, Tensor> =
                iter::once((0, xs.shallow_clone())).collect();
            let mut exported_tensors = vec![];

            // run the network
            self.layers.iter().for_each(|layer| {
                let Layer {
                    layer_index,
                    ref module,
                    ref input_indexes,
                    ref anchors_opt,
                } = *layer;

                let inputs: Vec<_> = input_indexes
                    .iter()
                    .map(|from_index| &tmp_tensors[from_index])
                    .collect();

                let output = module.forward_t(inputs.as_slice(), train);

                if let Some(_anchors) = anchors_opt {
                    exported_tensors.push(output.shallow_clone());
                }
                tmp_tensors.insert(layer_index, output);
            });

            // run detection module
            let exported_tensors: Vec<_> = exported_tensors.iter().collect();
            self.detection_module
                .forward_t(exported_tensors.as_slice(), train, &image_size)
        }
    }
}

mod yolo_output {
    use super::*;

    #[derive(Debug, CopyGetters, Getters, TensorLike)]
    pub struct YoloOutput {
        #[getset(get = "pub")]
        pub(crate) image_size: PixelSize<i64>,
        #[getset(get_copy = "pub")]
        pub(crate) batch_size: i64,
        #[getset(get_copy = "pub")]
        pub(crate) num_classes: i64,
        #[tensor_like(copy)]
        #[getset(get_copy = "pub")]
        pub(crate) device: Device,
        #[getset(get = "pub")]
        pub(crate) layer_meta: Vec<LayerMeta>,
        // below tensors have shape [batch, entry, flat] where
        // - flat = \sum_{i is layer_index} (n_anchors_i * feature_height_i * feature_width_i)
        #[getset(get = "pub")]
        pub(crate) cy: Tensor,
        #[getset(get = "pub")]
        pub(crate) cx: Tensor,
        #[getset(get = "pub")]
        pub(crate) height: Tensor,
        #[getset(get = "pub")]
        pub(crate) width: Tensor,
        #[getset(get = "pub")]
        pub(crate) objectness: Tensor,
        #[getset(get = "pub")]
        pub(crate) classification: Tensor,
    }

    impl YoloOutput {
        pub fn cat<T>(outputs: impl IntoIterator<Item = T>, device: Device) -> Result<Self>
        where
            T: Borrow<Self>,
        {
            let (
                image_size_set,
                batch_size_vec,
                num_classes_set,
                layer_meta_set,
                cy_vec,
                cx_vec,
                height_vec,
                width_vec,
                objectness_vec,
                classification_vec,
            ): (
                HashSet<PixelSize<i64>>,
                Vec<i64>,
                HashSet<i64>,
                HashSet<Vec<LayerMeta>>,
                Vec<Tensor>,
                Vec<Tensor>,
                Vec<Tensor>,
                Vec<Tensor>,
                Vec<Tensor>,
                Vec<Tensor>,
            ) = outputs
                .into_iter()
                .map(|output| {
                    let YoloOutput {
                        ref image_size,
                        batch_size,
                        num_classes,
                        ref layer_meta,
                        ref cy,
                        ref cx,
                        ref height,
                        ref width,
                        ref objectness,
                        ref classification,
                        ..
                    } = *output.borrow();

                    (
                        image_size.clone(),
                        batch_size,
                        num_classes,
                        layer_meta.to_owned(),
                        cy.to_device(device),
                        cx.to_device(device),
                        height.to_device(device),
                        width.to_device(device),
                        objectness.to_device(device),
                        classification.to_device(device),
                    )
                })
                .unzip_n();

            let image_size = {
                ensure!(image_size_set.len() == 1, "image_size must be equal");
                image_size_set.into_iter().next().unwrap()
            };
            let num_classes = {
                ensure!(num_classes_set.len() == 1, "num_classes must be equal");
                num_classes_set.into_iter().next().unwrap()
            };
            let layer_meta = {
                ensure!(layer_meta_set.len() == 1, "layer_meta must be equal");
                layer_meta_set.into_iter().next().unwrap()
            };
            let batch_size: i64 = batch_size_vec.into_iter().sum();
            let cy = Tensor::cat(&cy_vec, 0);
            let cx = Tensor::cat(&cx_vec, 0);
            let height = Tensor::cat(&height_vec, 0);
            let width = Tensor::cat(&width_vec, 0);
            let objectness = Tensor::cat(&objectness_vec, 0);
            let classification = Tensor::cat(&classification_vec, 0);

            let flat_index_size: i64 = layer_meta
                .iter()
                .map(|meta| {
                    let LayerMeta {
                        feature_size: GridSize { height, width, .. },
                        ref anchors,
                        ..
                    } = *meta;
                    height * width * anchors.len() as i64
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
                height.size3()? == (batch_size, 1, flat_index_size),
                "invalid height shape"
            );
            ensure!(
                width.size3()? == (batch_size, 1, flat_index_size),
                "invalid width shape"
            );
            ensure!(
                objectness.size3()? == (batch_size, 1, flat_index_size),
                "invalid objectness shape"
            );
            ensure!(
                classification.size3()? == (batch_size, num_classes, flat_index_size),
                "invalid classification shape"
            );

            Ok(Self {
                device,
                image_size,
                num_classes,
                layer_meta,
                batch_size,
                cy,
                cx,
                height,
                width,
                objectness,
                classification,
            })
        }

        pub fn flat_to_instance_index(
            &self,
            batch_index: usize,
            flat_index: i64,
        ) -> Option<InstanceIndex> {
            let Self { batch_size, .. } = *self;

            if batch_index as i64 >= batch_size || flat_index < 0 {
                return None;
            }

            let (
                layer_index,
                LayerMeta {
                    feature_size:
                        GridSize {
                            height: feature_h,
                            width: feature_w,
                            ..
                        },
                    anchors,
                    flat_index_range,
                    ..
                },
            ) = self
                .layer_meta
                .iter()
                .enumerate()
                .find(|(_layer_index, meta)| flat_index < meta.flat_index_range.end)?;

            // flat_index = begin_flat_index + col + row * (width + anchor_index * height)
            let remainder = flat_index - flat_index_range.start;
            let grid_col = remainder % feature_w;
            let grid_row = remainder / feature_w % feature_h;
            let anchor_index = remainder / feature_w / feature_h;

            if anchor_index >= anchors.len() as i64 {
                return None;
            }

            Some(InstanceIndex {
                batch_index,
                layer_index,
                anchor_index,
                grid_row,
                grid_col,
            })
        }

        pub fn instance_to_flat_index(&self, instance_index: &InstanceIndex) -> i64 {
            let InstanceIndex {
                layer_index,
                anchor_index,
                grid_row,
                grid_col,
                ..
            } = *instance_index;

            let LayerMeta {
                ref flat_index_range,
                feature_size: GridSize { height, width, .. },
                ..
            } = self.layer_meta[layer_index];

            let flat_index =
                flat_index_range.start + grid_col + width * (grid_row + height * anchor_index);

            flat_index
        }

        pub fn feature_maps(&self) -> Vec<FeatureMap> {
            let Self {
                batch_size,
                num_classes,
                ref layer_meta,
                ..
            } = *self;

            let feature_maps = layer_meta
                .iter()
                .enumerate()
                .map(|(_layer_index, meta)| {
                    let LayerMeta {
                        feature_size:
                            GridSize {
                                height: feature_h,
                                width: feature_w,
                                ..
                            },
                        ref anchors,
                        ref flat_index_range,
                        ..
                    } = *meta;
                    let num_anchors = anchors.len() as i64;

                    let cy_map = self.cy.i((.., .., flat_index_range.clone())).view([
                        batch_size,
                        1,
                        num_anchors,
                        feature_h,
                        feature_w,
                    ]);
                    let cx_map = self.cx.i((.., .., flat_index_range.clone())).view([
                        batch_size,
                        1,
                        num_anchors,
                        feature_h,
                        feature_w,
                    ]);
                    let h_map = self.height.i((.., .., flat_index_range.clone())).view([
                        batch_size,
                        1,
                        num_anchors,
                        feature_h,
                        feature_w,
                    ]);
                    let w_map = self.width.i((.., .., flat_index_range.clone())).view([
                        batch_size,
                        1,
                        num_anchors,
                        feature_h,
                        feature_w,
                    ]);
                    let objectness_map = self
                        .objectness
                        .i((.., .., flat_index_range.clone()))
                        .view([batch_size, 1, num_anchors, feature_h, feature_w]);
                    let classification_map = self
                        .classification
                        .i((.., .., flat_index_range.clone()))
                        .view([batch_size, num_classes, num_anchors, feature_h, feature_w]);

                    FeatureMap {
                        cy: cy_map,
                        cx: cx_map,
                        h: h_map,
                        w: w_map,
                        objectness: objectness_map,
                        classification: classification_map,
                    }
                })
                .collect_vec();

            feature_maps
        }
    }
}

mod misc {
    use super::*;

    #[derive(Debug)]
    pub struct Layer {
        pub(crate) layer_index: usize,
        pub(crate) module: YoloModule,
        pub(crate) input_indexes: Vec<usize>,
        pub(crate) anchors_opt: Option<Vec<(usize, usize)>>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, TensorLike)]
    pub struct LayerMeta {
        /// feature map size in grid units
        #[tensor_like(clone)]
        pub feature_size: GridSize<i64>,
        /// per grid size in pixel units
        #[tensor_like(clone)]
        pub grid_size: PixelSize<R64>,
        /// Anchros (height, width) in grid units
        #[tensor_like(clone)]
        pub anchors: Vec<GridSize<R64>>,
        #[tensor_like(clone)]
        pub flat_index_range: Range<i64>,
    }

    #[derive(Debug, TensorLike)]
    pub struct FeatureMap {
        // tensors have shape [batch, entry, anchor, height, width]
        pub cy: Tensor,
        pub cx: Tensor,
        pub h: Tensor,
        pub w: Tensor,
        pub objectness: Tensor,
        pub classification: Tensor,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, TensorLike)]
    pub struct InstanceIndex {
        pub batch_index: usize,
        pub layer_index: usize,
        pub anchor_index: i64,
        pub grid_row: i64,
        pub grid_col: i64,
    }
}
