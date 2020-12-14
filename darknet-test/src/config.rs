use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    pub config_file: PathBuf,
    pub weights_file: PathBuf,
    // #[serde(with = "tch_serde::serde_device")]
    // pub darknet_device: Device,
    #[serde(with = "tch_serde::serde_device")]
    pub rust_device: Device,
}
