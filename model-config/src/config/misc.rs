use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Shape(Vec<usize>);

impl From<Vec<usize>> for Shape {
    fn from(vec: Vec<usize>) -> Self {
        Self(vec)
    }
}

impl From<&[usize]> for Shape {
    fn from(slice: &[usize]) -> Self {
        Self(Vec::from(slice))
    }
}

impl From<Shape> for Vec<usize> {
    fn from(shape: Shape) -> Self {
        shape.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "(R64, R64)", into = "(R64, R64)")]
pub struct Size {
    pub h: R64,
    pub w: R64,
}

impl From<(R64, R64)> for Size {
    fn from((h, w): (R64, R64)) -> Self {
        Self { h, w }
    }
}

impl From<Size> for (R64, R64) {
    fn from(Size { h, w }: Size) -> Self {
        (h, w)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Activation {
    #[serde(rename = "mish")]
    Mish,
    #[serde(rename = "hard_mish")]
    HardMish,
    #[serde(rename = "swish")]
    Swish,
    #[serde(rename = "normalize_channels")]
    NormalizeChannels,
    #[serde(rename = "normalize_channels_softmax")]
    NormalizeChannelsSoftmax,
    #[serde(rename = "normalize_channels_softmax_maxval")]
    NormalizeChannelsSoftmaxMaxval,
    #[serde(rename = "logistic")]
    Logistic,
    #[serde(rename = "loggy")]
    Loggy,
    #[serde(rename = "relu")]
    Relu,
    #[serde(rename = "elu")]
    Elu,
    #[serde(rename = "selu")]
    Selu,
    #[serde(rename = "gelu")]
    Gelu,
    #[serde(rename = "relie")]
    Relie,
    #[serde(rename = "ramp")]
    Ramp,
    #[serde(rename = "linear")]
    Linear,
    #[serde(rename = "tanh")]
    Tanh,
    #[serde(rename = "plse")]
    Plse,
    #[serde(rename = "leaky")]
    Leaky,
    #[serde(rename = "stair")]
    Stair,
    #[serde(rename = "hardtan")]
    Hardtan,
    #[serde(rename = "lhtan")]
    Lhtan,
    #[serde(rename = "relu6")]
    Relu6,
}
