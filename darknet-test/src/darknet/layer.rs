use super::kinds::{
    Activation, CostType, IoULoss, LayerType, NmsKind, WeightsNormalizion, WeightsType, YoloPoint,
};
use crate::{common::*, sys};

/// A layer of the network.
#[derive(Debug)]
#[repr(transparent)]
pub struct Layer {
    pub(crate) layer: sys::layer,
}

impl Layer {
    /// Get layer index in network.
    pub fn index(&self) -> usize {
        self.layer.index as usize
    }

    /// Get the type of layer.
    pub fn type_(&self) -> Option<LayerType> {
        FromPrimitive::from_usize(self.layer.type_ as usize)
    }

    /// Get the activation type.
    pub fn activation(&self) -> Option<Activation> {
        FromPrimitive::from_usize(self.layer.activation as usize)
    }

    /// Get the cost (or namely the loss) type.
    pub fn cost_type(&self) -> Option<CostType> {
        FromPrimitive::from_usize(self.layer.activation as usize)
    }

    /// Get the weights format type.
    pub fn weights_type(&self) -> Option<WeightsType> {
        FromPrimitive::from_usize(self.layer.weights_type as usize)
    }

    /// Get the weights normalization type.
    pub fn weights_normalization(&self) -> Option<WeightsNormalizion> {
        FromPrimitive::from_usize(self.layer.weights_normalization as usize)
    }

    /// Get the non-maximum suppression (NMS) type.
    pub fn nms_kind(&self) -> Option<NmsKind> {
        FromPrimitive::from_usize(self.layer.nms_kind as usize)
    }

    /// Get the YOLO point type.
    pub fn yolo_point(&self) -> Option<YoloPoint> {
        FromPrimitive::from_usize(self.layer.yolo_point as usize)
    }

    /// Get the IoU loss type.
    pub fn iou_loss(&self) -> Option<IoULoss> {
        FromPrimitive::from_usize(self.layer.iou_loss as usize)
    }

    /// Get the IoU threshold type.
    pub fn iou_thresh_kind(&self) -> Option<IoULoss> {
        FromPrimitive::from_usize(self.layer.iou_thresh_kind as usize)
    }

    /// Get the input tensor height.
    pub fn input_height(&self) -> usize {
        self.layer.h as usize
    }

    /// Get the input tensor width.
    pub fn input_width(&self) -> usize {
        self.layer.w as usize
    }

    /// Get the input tensor channels.
    pub fn input_channels(&self) -> usize {
        self.layer.c as usize
    }

    /// Get the input shape tuple (width, height, channels).
    pub fn input_shape(&self) -> (usize, usize, usize) {
        (
            self.input_width(),
            self.input_height(),
            self.input_channels(),
        )
    }

    /// Get the output tensor height.
    pub fn output_height(&self) -> usize {
        self.layer.out_h as usize
    }

    /// Get the output tensor width.
    pub fn output_width(&self) -> usize {
        self.layer.out_w as usize
    }

    /// Get the output tensor channels.
    pub fn output_channels(&self) -> usize {
        self.layer.out_c as usize
    }

    /// Get the output shape tuple (width, height, channels).
    pub fn output_shape(&self) -> (usize, usize, usize) {
        (
            self.output_width(),
            self.output_height(),
            self.output_channels(),
        )
    }

    pub fn batch(&self) -> usize {
        self.layer.batch as usize
    }

    pub fn output_slice(&self) -> &[f32] {
        unsafe {
            let batch = self.batch();
            let (out_w, out_h, out_c) = self.output_shape();
            let output_size = batch * out_w * out_h * out_c;
            let output_ptr = sys::layer_get_output(&self.layer as *const _);
            let slice = slice::from_raw_parts_mut(output_ptr, output_size);
            slice
        }
    }

    pub fn output_array(&self) -> ArrayView4<'_, f32> {
        unsafe {
            let batch = self.batch();
            let (out_w, out_h, out_c) = self.output_shape();
            let output_ptr = sys::layer_get_output(&self.layer as *const _);
            ArrayView4::from_shape_ptr((batch, out_w, out_h, out_c), output_ptr)
        }
    }
}

impl Deref for Layer {
    type Target = sys::layer;

    fn deref(&self) -> &Self::Target {
        &self.layer
    }
}
