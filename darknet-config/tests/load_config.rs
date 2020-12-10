use anyhow::{Context, Result};
use darknet_config::DarknetConfig;
use std::path::Path;

#[test]
fn load_darknet_config() -> Result<()> {
    glob::glob(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("cfg")
            .join("*.cfg")
            .to_str()
            .unwrap(),
    )?
    .try_for_each(|path| -> Result<_> {
        let path = path?;
        let config = DarknetConfig::load(&path)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(())
    })?;

    Ok(())
}
