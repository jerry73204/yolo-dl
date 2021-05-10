use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Connected {
    #[serde(default = "defaults::connected_output")]
    pub output: usize,
    #[serde(default = "defaults::connected_activation")]
    pub activation: Activation,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub batch_normalize: bool,
    #[serde(flatten)]
    pub common: Common,
}

impl Connected {
    pub fn output_shape(&self, input_shape: [usize; 1]) -> Option<[usize; 1]> {
        (input_shape[0] > 0).then(|| [self.output])
    }
}
