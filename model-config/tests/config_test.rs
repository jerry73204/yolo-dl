use anyhow::{Context, Result};
use model_config::config::Model;
use std::{
    fs,
    path::{Path, PathBuf},
};

const MODEL_FILE_NAMES: &[&str] = &["model.json5"];

lazy_static::lazy_static! {
    static ref CONFIG_DIR: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("test").join("cfg");
    static ref MODEL_FILES: Vec<PathBuf> = {
        MODEL_FILE_NAMES.iter().map(|file_name| CONFIG_DIR.join(file_name)).collect()
    };
}

#[test]
fn model_config_test() -> Result<()> {
    MODEL_FILES.iter().try_for_each(|path| -> Result<_> {
        let _: Model = json5::from_str(
            &fs::read_to_string(path)
                .with_context(|| format!("failed to read '{}'", path.display()))?,
        )
        .with_context(|| format!("failed to parse '{}'", path.display()))?;
        Ok(())
    })?;
    Ok(())
}
