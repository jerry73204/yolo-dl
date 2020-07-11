use crate::common::*;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub dataset_dir: PathBuf,
    pub dataset_name: String,
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
