use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Cost {
    #[serde(default = "cost_sse")]
    pub r#type: CostType,
    #[serde(default = "num_traits::one")]
    pub scale: R64,
    #[serde(default = "num_traits::zero")]
    pub ratio: R64,
    #[serde(flatten)]
    pub common: Meta,
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

fn cost_sse() -> CostType {
    CostType::Sse
}
