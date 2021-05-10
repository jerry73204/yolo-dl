use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Crop {
    #[serde(default = "defaults::crop_stride")]
    pub stride: usize,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub reverse: bool,
    #[serde(flatten)]
    pub common: Common,
}
