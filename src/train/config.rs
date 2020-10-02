use crate::common::*;
use yolo_dl::{
    loss::{IoUKind, MatchGrid},
    utils::Ratio,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub dataset_dir: PathBuf,
    pub dataset_name: String,
    pub whitelist_classes: Option<Vec<String>>,
    pub cache_dir: PathBuf,
    pub image_size: NonZeroUsize,
    pub mosaic_prob: Ratio,
    pub mosaic_margin: Ratio,
    pub affine_prob: Ratio,
    pub logging_dir: PathBuf,
    pub log_images: bool,
    pub mini_batch_size: usize,
    pub rotate_degrees: Option<R64>,
    pub translation: Option<R64>,
    pub scale: Option<(R64, R64)>,
    pub shear: Option<R64>,
    pub horizontal_flip: bool,
    pub vertical_flip: bool,
    pub match_grid_method: MatchGrid,
    pub iou_kind: IoUKind,
    #[serde(with = "tch_serde::serde_device")]
    pub device: Device,
}

impl Config {
    pub fn open<P>(path: P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let text = std::fs::read_to_string(path)?;
        let config = json5::from_str(&text)?;
        Ok(config)
    }
}
