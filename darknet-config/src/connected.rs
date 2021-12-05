use super::{Activation, Meta};
use crate::{common::*, utils};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Connected {
    #[serde(default = "num_traits::one")]
    pub output: usize,
    #[serde(default = "default_activation")]
    pub activation: Activation,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub batch_normalize: bool,
    #[serde(flatten)]
    pub common: Meta,
}

impl Connected {
    pub fn output_shape(&self, input_shape: [usize; 1]) -> Option<[usize; 1]> {
        (input_shape[0] > 0).then(|| [self.output])
    }
}

fn default_activation() -> Activation {
    Activation::Logistic
}
