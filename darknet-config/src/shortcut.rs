use super::{Activation, Meta};
use crate::{common::*, utils::FromLayers};

#[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
#[derivative(Hash)]
pub struct Shortcut {
    pub from: FromLayers,
    pub activation: Activation,
    #[serde(with = "serde_weights_type", default = "weights_type_none")]
    pub weights_type: WeightsType,
    #[serde(default = "default_weights_normalization")]
    pub weights_normalization: WeightsNormalization,
    #[serde(flatten)]
    pub common: Meta,
}

impl Shortcut {
    pub fn output_shape(&self, input_shapes: &[[usize; 3]]) -> Option<[usize; 3]> {
        let set: HashSet<_> = input_shapes.iter().map(|&[h, w, _c]| [h, w]).collect();
        (set.len() == 1).then(|| input_shapes[0])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeightsType {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "per_feature")]
    PerFeature,
    #[serde(rename = "per_channel")]
    PerChannel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeightsNormalization {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "relu")]
    Relu,
    #[serde(rename = "softmax")]
    Softmax,
}

mod serde_weights_type {
    use super::*;

    pub fn serialize<S>(weights_type: &WeightsType, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        weights_type.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<WeightsType, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        let weights_type = match text.as_str() {
            "per_feature" | "per_layer" => WeightsType::PerFeature,
            "per_channel" => WeightsType::PerChannel,
            _ => {
                return Err(D::Error::custom(format!(
                    "'{}' is not a valid weights type",
                    text
                )))
            }
        };
        Ok(weights_type)
    }
}

fn default_weights_normalization() -> WeightsNormalization {
    WeightsNormalization::None
}

fn weights_type_none() -> WeightsType {
    WeightsType::None
}
