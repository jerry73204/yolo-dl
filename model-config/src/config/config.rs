use super::model::Layer;
use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Config {
    pub input_channels: usize,
    pub num_classes: usize,
    pub depth_multiple: R64,
    pub width_multiple: R64,
    pub layers: Vec<Layer>,
}
