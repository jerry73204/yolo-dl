use super::Meta;
use crate::{common::*, utils};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Crop {
    #[serde(default = "num_traits::one")]
    pub stride: usize,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub reverse: bool,
    #[serde(flatten)]
    pub common: Meta,
}
