use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Cost {
    #[serde(default = "defaults::cost_type")]
    pub r#type: CostType,
    #[serde(default = "defaults::cost_scale")]
    pub scale: R64,
    #[serde(default = "defaults::cost_ratio")]
    pub ratio: R64,
    #[serde(flatten)]
    pub common: Common,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CostType {
    #[serde(rename = "sse")]
    Sse,
    #[serde(rename = "masked")]
    Masked,
    #[serde(rename = "smooth")]
    Smooth,
}
