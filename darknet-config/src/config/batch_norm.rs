use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BatchNorm {
    #[serde(flatten)]
    pub common: Common,
}
