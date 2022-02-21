use crate::common::*;

pub use activation::*;
mod activation {
    use super::*;

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
}

pub use layer_index::*;
mod layer_index {
    use super::*;

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

        pub fn from_ordinal(index: isize) -> Self {
            if index < 0 {
                Self::Relative(NonZeroUsize::new(-index as usize).unwrap())
            } else {
                Self::Absolute(index as usize)
            }
        }

        pub fn ordinal(&self) -> isize {
            match *self {
                Self::Relative(val) => -(val.get() as isize),
                Self::Absolute(val) => val as isize,
            }
        }
    }

    impl Display for LayerIndex {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            let text = match *self {
                LayerIndex::Relative(index) => {
                    format!("-{}", index)
                }
                LayerIndex::Absolute(index) => index.to_string(),
            };
            formatter.write_str(&text)
        }
    }

    // impl From<isize> for LayerIndex {
    //     fn from(index: isize) -> Self {
    //         if index < 0 {
    //             Self::Relative(NonZeroUsize::new(-index as usize).unwrap())
    //         } else {
    //             Self::Absolute(index as usize)
    //         }
    //     }
    // }

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
            let index = isize::deserialize(deserializer)?;
            Ok(Self::from_ordinal(index))
        }
    }
}
