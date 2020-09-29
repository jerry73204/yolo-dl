use super::*;
use crate::{
    common::*,
    utils::{GridSize, PixelSize, Unzip6},
};

#[derive(Debug)]
pub struct YoloModel {
    pub(crate) layers: Vec<Layer>,
    pub(crate) detection_module: DetectModule,
}

impl YoloModel {
    pub fn forward_t(&self, xs: &Tensor, train: bool) -> YoloOutput {
        let (_batch_size, _channels, height, width) = xs.size4().unwrap();
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
            .forward_t(exported_tensors.as_slice(), train, height, width)
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
    pub(crate) outputs: Vec<LayerOutput>,
}

impl YoloOutput {
    pub fn get(&self, index: &InstanceIndex) -> Option<Instance> {
        let InstanceIndex {
            batch_index,
            layer_index,
            anchor_index,
            grid_row,
            grid_col,
        } = *index;
        let batch_index = batch_index as i64;

        let layer = self.outputs.get(layer_index)?;
        let cy = layer
            .cy
            .i((batch_index, anchor_index, grid_row, grid_col, ..));
        let cx = layer
            .cx
            .i((batch_index, anchor_index, grid_row, grid_col, ..));
        let height = layer
            .height
            .i((batch_index, anchor_index, grid_row, grid_col, ..));
        let width = layer
            .width
            .i((batch_index, anchor_index, grid_row, grid_col, ..));
        let objectness = layer
            .objectness
            .i((batch_index, anchor_index, grid_row, grid_col, ..));
        let classification =
            layer
                .classification
                .i((batch_index, anchor_index, grid_row, grid_col, ..));

        Some(Instance {
            cy,
            cx,
            height,
            width,
            objectness,
            classification,
        })
    }
}

#[derive(Debug, TensorLike)]
pub struct FeatureInfo {
    /// feature map size in grid units
    pub feature_size: GridSize<i64>,
    /// per grid size in pixel units
    pub grid_size: PixelSize<f64>,
    /// Anchros (height, width) in grid units
    pub anchors: Vec<GridSize<f64>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, TensorLike)]
pub struct DetectionIndex {
    pub layer_index: usize,
    pub anchor_index: i64,
    pub grid_row: i64,
    pub grid_col: i64,
}

#[derive(Debug, TensorLike)]
pub struct Detection {
    pub index: DetectionIndex,
    pub cycxhw: Tensor,
    pub objectness: Tensor,
    pub classification: Tensor,
}

#[derive(Debug, TensorLike)]
pub struct LayerOutput {
    pub cy: Tensor,
    pub cx: Tensor,
    pub height: Tensor,
    pub width: Tensor,
    pub objectness: Tensor,
    pub classification: Tensor,
    /// feature map size in grid units
    pub feature_size: GridSize<i64>,
    /// per grid size in pixel units
    pub grid_size: PixelSize<f64>,
    /// Anchros (height, width) in grid units
    pub anchors: Vec<GridSize<f64>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, TensorLike)]
pub struct InstanceIndex {
    pub batch_index: usize,
    pub layer_index: usize,
    pub anchor_index: i64,
    pub grid_row: i64,
    pub grid_col: i64,
}

#[derive(Debug, TensorLike)]
pub struct MultiInstance {
    pub cys: Tensor,
    pub cxs: Tensor,
    pub heights: Tensor,
    pub widths: Tensor,
    pub objectnesses: Tensor,
    pub classifications: Tensor,
}

impl MultiInstance {
    pub fn new(instances: &[Instance]) -> MultiInstance {
        let (cys, cxs, heights, widths, objectnesses, classifications) = instances
            .iter()
            .map(|instance| {
                let Instance {
                    cy,
                    cx,
                    height,
                    width,
                    objectness,
                    classification,
                } = instance;
                (cy, cx, height, width, objectness, classification)
            })
            .unzip_n_vec();

        MultiInstance {
            cys: Tensor::stack(&cys, 0),
            cxs: Tensor::stack(&cxs, 0),
            heights: Tensor::stack(&heights, 0),
            widths: Tensor::stack(&widths, 0),
            objectnesses: Tensor::stack(&objectnesses, 0),
            classifications: Tensor::stack(&classifications, 0),
        }
    }
}

#[derive(Debug, TensorLike)]
pub struct Instance {
    pub cy: Tensor,
    pub cx: Tensor,
    pub height: Tensor,
    pub width: Tensor,
    pub objectness: Tensor,
    pub classification: Tensor,
}
