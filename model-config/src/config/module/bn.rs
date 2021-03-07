use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BatchNorm {
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default = "default_affine")]
    pub affine: bool,
    pub var_min: Option<R64>,
    pub var_max: Option<R64>,
}

impl Default for BatchNorm {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            affine: default_affine(),
            var_min: None,
            var_max: None,
        }
    }
}

fn default_enabled() -> bool {
    true
}

fn default_affine() -> bool {
    true
}
