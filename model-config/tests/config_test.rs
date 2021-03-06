use anyhow::Result;
use model_config::Model;

#[test]
fn model_config_test() -> Result<()> {
    for file in glob::glob(&format!("{}/cfg/model/*.json", env!("CARGO_MANIFEST_DIR")))? {
        let _ = Model::load(file?)?;
    }

    Ok(())
}
