use super::Meta;
use crate::{common::*, utils};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "RawMaxPool", into = "RawMaxPool")]
pub struct MaxPool {
    pub stride_x: usize,
    pub stride_y: usize,
    pub size: usize,
    pub padding: usize,
    pub maxpool_depth: bool,
    pub out_channels: usize,
    pub antialiasing: bool,
    #[serde(flatten)]
    pub common: Meta,
}

impl MaxPool {
    pub fn output_shape(&self, input_shape: [usize; 3]) -> [usize; 3] {
        let Self {
            padding,
            size,
            stride_x,
            stride_y,
            ..
        } = *self;
        let [in_h, in_w, in_c] = input_shape;

        let out_h = (in_h + padding - size) / stride_y + 1;
        let out_w = (in_w + padding - size) / stride_x + 1;
        let out_c = in_c;

        [out_h, out_w, out_c]
    }
}

impl From<RawMaxPool> for MaxPool {
    fn from(raw: RawMaxPool) -> Self {
        let RawMaxPool {
            stride,
            stride_x,
            stride_y,
            size,
            padding,
            maxpool_depth,
            out_channels,
            antialiasing,
            common,
        } = raw;

        let stride_x = stride_x.unwrap_or(stride);
        let stride_y = stride_y.unwrap_or(stride);
        let size = size.unwrap_or(stride);
        let padding = padding.unwrap_or(size - 1);

        Self {
            stride_x,
            stride_y,
            size,
            padding,
            maxpool_depth,
            out_channels,
            antialiasing,
            common,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct RawMaxPool {
    #[serde(default = "num_traits::one")]
    pub stride: usize,
    pub stride_x: Option<usize>,
    pub stride_y: Option<usize>,
    pub size: Option<usize>,
    pub padding: Option<usize>,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub maxpool_depth: bool,
    #[serde(default = "num_traits::one")]
    pub out_channels: usize,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub antialiasing: bool,
    #[serde(flatten)]
    pub common: Meta,
}

impl From<MaxPool> for RawMaxPool {
    fn from(maxpool: MaxPool) -> Self {
        let MaxPool {
            stride_x,
            stride_y,
            size,
            padding,
            maxpool_depth,
            out_channels,
            antialiasing,
            common,
        } = maxpool;

        Self {
            stride: num_traits::one(),
            stride_x: Some(stride_x),
            stride_y: Some(stride_y),
            size: Some(size),
            padding: Some(padding),
            maxpool_depth,
            out_channels,
            antialiasing,
            common,
        }
    }
}
