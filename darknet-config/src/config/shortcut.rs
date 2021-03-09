use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
#[derivative(Hash)]
pub struct Shortcut {
    #[derivative(Hash(hash_with = "hash_vec::<LayerIndex, _>"))]
    #[serde(with = "serde_::vec_layers")]
    pub from: IndexSet<LayerIndex>,
    pub activation: Activation,
    #[serde(with = "serde_::weights_type", default = "defaults::weights_type")]
    pub weights_type: WeightsType,
    #[serde(default = "defaults::weights_normalization")]
    pub weights_normalization: WeightsNormalization,
    #[serde(flatten)]
    pub common: Common,
}
