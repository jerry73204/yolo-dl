#[cfg(feature = "tch")]
pub use impls::*;
#[cfg(feature = "tch")]
mod impls;

#[cfg(feature = "tch")]
pub use r#trait::*;
#[cfg(feature = "tch")]
mod r#trait;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
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
