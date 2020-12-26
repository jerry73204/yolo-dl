use anyhow::Result;
use model_config::config::Model;
use std::path::{Path, PathBuf};

const MODEL_FILE_NAMES: &[&str] = &["yolov4-csp.json5"];

lazy_static::lazy_static! {
    static ref CONFIG_DIR: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("cfg");
    static ref MODEL_FILES: Vec<PathBuf> = {
        MODEL_FILE_NAMES.iter().map(|file_name| CONFIG_DIR.join(file_name)).collect()
    };
}

#[test]
fn model_config_test() -> Result<()> {
    MODEL_FILES.iter().try_for_each(|path| -> Result<_> {
        Model::load(path)?;
        Ok(())
    })?;
    Ok(())
}
