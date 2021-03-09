use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AvgPool {
    #[serde(flatten)]
    pub common: Common,
}
