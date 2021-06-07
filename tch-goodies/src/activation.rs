use crate::common::*;

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
