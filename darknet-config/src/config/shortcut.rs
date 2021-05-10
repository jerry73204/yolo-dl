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

impl Shortcut {
    pub fn output_shape(&self, input_shapes: &[[usize; 3]]) -> Option<[usize; 3]> {
        let set: HashSet<_> = input_shapes.iter().map(|&[h, w, _c]| [h, w]).collect();
        (set.len() == 1).then(|| input_shapes[0])
    }
}
