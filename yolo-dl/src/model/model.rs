use super::*;
use crate::common::*;

#[derive(Debug)]
pub struct YoloModel {
    pub(crate) layers: Vec<Layer>,
    pub(crate) detection_module: DetectModule,
}

impl YoloModel {
    pub fn forward_t(&mut self, xs: &Tensor, train: bool) -> YoloOutput {
        let (_batch_size, _channels, height, width) = xs.size4().unwrap();
        let image_size = PixelSize::new(height, width);
        let mut tmp_tensors: HashMap<usize, Tensor> = iter::once((0, xs.shallow_clone())).collect();
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

#[derive(Debug)]
pub struct Layer {
    pub(crate) layer_index: usize,
    pub(crate) module: YoloModule,
    pub(crate) input_indexes: Vec<usize>,
    pub(crate) anchors_opt: Option<Vec<(usize, usize)>>,
}

#[derive(Debug, TensorLike)]
pub struct YoloOutput {
    pub(crate) image_size: PixelSize<i64>,
    pub(crate) batch_size: i64,
    pub(crate) num_classes: i64,
    #[tensor_like(copy)]
    pub(crate) device: Device,
    pub(crate) layer_meta: Vec<LayerMeta>,
    // below tensors have shape [n_instances, n_outputs] where
    // - n_instances = (\sum_(1<= i <= n_layers) batch_size x n_anchors_i x feature_height_i x feature_width_i)
    // - n_outputs depends on output kind (cy, cx, height, width, objectness -> 1; classification -> n_classes)
    pub(crate) cy: Tensor,
    pub(crate) cx: Tensor,
    pub(crate) height: Tensor,
    pub(crate) width: Tensor,
    pub(crate) objectness: Tensor,
    pub(crate) classification: Tensor,
}

impl YoloOutput {
    pub fn image_size(&self) -> &PixelSize<i64> {
        &self.image_size
    }

    pub fn layer_meta(&self) -> &[LayerMeta] {
        &self.layer_meta
    }

    pub fn cy(&self) -> &Tensor {
        &self.cy
    }

    pub fn cx(&self) -> &Tensor {
        &self.cx
    }

    pub fn height(&self) -> &Tensor {
        &self.height
    }

    pub fn width(&self) -> &Tensor {
        &self.width
    }

    pub fn classification(&self) -> &Tensor {
        &self.classification
    }

    pub fn objectness(&self) -> &Tensor {
        &self.objectness
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
            begin_flat_index,
            feature_size: GridSize { height, width, .. },
            ..
        } = self.layer_meta[layer_index];

        let flat_index = begin_flat_index + grid_col + width * (grid_row + height * anchor_index);

        flat_index
    }
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
    pub begin_flat_index: i64,
    pub end_flat_index: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, TensorLike)]
pub struct InstanceIndex {
    pub batch_index: usize,
    pub layer_index: usize,
    pub anchor_index: i64,
    pub grid_row: i64,
    pub grid_col: i64,
}
