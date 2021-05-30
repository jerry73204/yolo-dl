use anyhow::Result;
use model_graph::Graph;

#[test]
fn model_config_test() -> Result<()> {
    for file in glob::glob(&format!("{}/cfg/model/*.json", env!("CARGO_MANIFEST_DIR")))? {
        let _ = Graph::load_newslab_v1_json(file?)?;
    }

    Ok(())
}
