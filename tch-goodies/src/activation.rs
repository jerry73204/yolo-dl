use crate::{common::*, tensor::TensorExt};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Mish,
    HardMish,
    Swish,
    NormalizeChannels,
    NormalizeChannelsSoftmax,
    NormalizeChannelsSoftmaxMaxval,
    Logistic,
    Loggy,
    Relu,
    LRelu,
    Elu,
    Selu,
    Gelu,
    Relie,
    Ramp,
    Linear,
    Tanh,
    Plse,
    Leaky,
    Stair,
    Hardtan,
    Lhtan,
    Relu6,
}

impl nn::Module for Activation {
    fn forward(&self, xs: &Tensor) -> Tensor {
        use Activation::*;

        match *self {
            Linear => xs.shallow_clone(),
            Mish => xs.mish(),
            HardMish => xs.hard_mish(),
            Swish => xs.swish(),
            Relu => xs.relu(),
            Leaky => xs.clamp_min(0.0) + xs.clamp_max(0.0) * 0.1,
            Logistic => xs.sigmoid(),
            LRelu => xs.lrelu(),
            Elu => xs.elu(),
            Selu => xs.selu(),
            Gelu => xs.gelu(),
            Tanh => xs.tanh(),
            Hardtan => xs.hardtanh(),
            _ => todo!(),
        }
    }
}
