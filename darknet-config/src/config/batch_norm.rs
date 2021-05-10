use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BatchNorm {
    #[serde(flatten)]
    pub common: Common,
}

impl BatchNorm {
    pub fn output_shape(&self, input_shape: Shape) -> Shape {
        input_shape
    }
}
