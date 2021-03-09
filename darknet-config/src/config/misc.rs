use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Deform {
    None,
    Sway,
    Rotate,
    Stretch,
    StretchSway,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IouLoss {
    #[serde(rename = "mse")]
    Mse,
    #[serde(rename = "iou")]
    IoU,
    #[serde(rename = "giou")]
    GIoU,
    #[serde(rename = "diou")]
    DIoU,
    #[serde(rename = "ciou")]
    CIoU,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IouThreshold {
    #[serde(rename = "iou")]
    IoU,
    #[serde(rename = "giou")]
    GIoU,
    #[serde(rename = "diou")]
    DIoU,
    #[serde(rename = "ciou")]
    CIoU,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum YoloPoint {
    #[serde(rename = "center")]
    Center,
    #[serde(rename = "left_top")]
    LeftTop,
    #[serde(rename = "right_bottom")]
    RightBottom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NmsKind {
    #[serde(rename = "default")]
    Default,
    #[serde(rename = "greedynms")]
    Greedy,
    #[serde(rename = "diounms")]
    DIoU,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PolicyKind {
    #[serde(rename = "random")]
    Random,
    #[serde(rename = "poly")]
    Poly,
    #[serde(rename = "constant")]
    Constant,
    #[serde(rename = "step")]
    Step,
    #[serde(rename = "exp")]
    Exp,
    #[serde(rename = "sigmoid")]
    Sigmoid,
    #[serde(rename = "steps")]
    Steps,
    #[serde(rename = "sgdr")]
    Sgdr,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Policy {
    Random,
    Poly,
    Constant,
    Step {
        step: u64,
        scale: R64,
    },
    Exp {
        gamma: R64,
    },
    Sigmoid {
        gamma: R64,
        step: u64,
    },
    Steps {
        steps: Vec<u64>,
        scales: Vec<R64>,
        seq_scales: Vec<R64>,
    },
    Sgdr,
    SgdrCustom {
        steps: Vec<u64>,
        scales: Vec<R64>,
        seq_scales: Vec<R64>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize_repr, Deserialize_repr)]
#[repr(u64)]
pub enum MixUp {
    MixUp = 1,
    CutMix = 2,
    Mosaic = 3,
    Random = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerIndex {
    Relative(NonZeroUsize),
    Absolute(usize),
}

impl LayerIndex {
    pub fn relative(&self) -> Option<usize> {
        match *self {
            Self::Relative(index) => Some(index.get()),
            Self::Absolute(_) => None,
        }
    }

    pub fn absolute(&self) -> Option<usize> {
        match *self {
            Self::Absolute(index) => Some(index),
            Self::Relative(_) => None,
        }
    }

    pub fn to_absolute(&self, curr_index: usize) -> Option<usize> {
        match *self {
            Self::Absolute(index) => Some(index),
            Self::Relative(index) => {
                let index = index.get();
                if index <= curr_index {
                    Some(curr_index - index)
                } else {
                    None
                }
            }
        }
    }
}

impl From<isize> for LayerIndex {
    fn from(index: isize) -> Self {
        if index < 0 {
            Self::Relative(NonZeroUsize::new(-index as usize).unwrap())
        } else {
            Self::Absolute(index as usize)
        }
    }
}

impl From<LayerIndex> for isize {
    fn from(index: LayerIndex) -> Self {
        match index {
            LayerIndex::Relative(index) => -(index.get() as isize),
            LayerIndex::Absolute(index) => index as isize,
        }
    }
}

impl Serialize for LayerIndex {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        isize::from(*self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for LayerIndex {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let index: Self = isize::deserialize(deserializer)?.into();
        Ok(index)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Shape {
    Hwc([u64; 3]),
    Flat(u64),
}

impl From<&Shape> for Vec<u64> {
    fn from(from: &Shape) -> Self {
        match *from {
            Shape::Hwc(hwc) => Vec::from(hwc),
            Shape::Flat(flat) => vec![flat],
        }
    }
}

impl Shape {
    pub fn iter(&self) -> impl Iterator<Item = u64> {
        Vec::<u64>::from(self).into_iter()
    }

    pub fn hwc(&self) -> Option<[u64; 3]> {
        match *self {
            Self::Hwc(hwc) => Some(hwc),
            Self::Flat(_) => None,
        }
    }

    pub fn flat(&self) -> Option<u64> {
        match *self {
            Self::Flat(flat) => Some(flat),
            Self::Hwc(_) => None,
        }
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hwc([h, w, c]) => f.debug_list().entries(vec![h, w, c]).finish(),
            Self::Flat(size) => write!(f, "{}", size),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Adam {
    pub b1: R64,
    pub b2: R64,
    pub eps: R64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeightsType {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "per_feature")]
    PerFeature,
    #[serde(rename = "per_channel")]
    PerChannel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeightsNormalization {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "relu")]
    Relu,
    #[serde(rename = "softmax")]
    Softmax,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RouteGroup {
    group_id: u64,
    num_groups: u64,
}

impl RouteGroup {
    pub fn new(group_id: u64, num_groups: u64) -> Option<Self> {
        if num_groups == 0 || group_id >= num_groups {
            None
        } else {
            Some(Self {
                group_id,
                num_groups,
            })
        }
    }

    pub fn group_id(&self) -> u64 {
        self.group_id
    }

    pub fn num_groups(&self) -> u64 {
        self.num_groups
    }
}
