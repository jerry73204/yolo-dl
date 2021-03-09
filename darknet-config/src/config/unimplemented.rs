use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UnimplementedLayer {
    #[serde(flatten)]
    pub common: Common,
}
