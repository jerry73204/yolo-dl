use crate::common::*;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub dataset_dir: PathBuf,
    pub dataset_name: String,
    pub cache_dir: PathBuf,
    pub image_size: NonZeroUsize,
    pub mosaic_margin: R64,
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
