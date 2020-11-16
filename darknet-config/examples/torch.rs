use anyhow::{Context, Result};
use argh::FromArgs;
use darknet_config::{DarknetConfig, DarknetModel, LayerBase, ModelBase, TchModel};
use std::path::PathBuf;
use tch::{nn, Device};

#[derive(Debug, Clone, FromArgs)]
/// Load darknet config and weights files and produce summary.
struct Args {
    #[argh(positional)]
    /// configuration file
    config_file: PathBuf,
    #[argh(positional)]
    /// weights file
    weights_file: PathBuf,
    #[argh(positional)]
    /// image file
    image_file: PathBuf,
}

fn main() -> Result<()> {
    pretty_env_logger::init();

    let Args {
        config_file,
        weights_file,
        image_file,
    } = argh::from_env();

    let vs = nn::VarStore::new(Device::Cpu);
    let root = vs.root();
    let model = {
        let mut darknet_model = DarknetModel::from_config_file(&config_file)?;
        darknet_model.load_weights(&weights_file)?;
        TchModel::from_darknet_model(&root, &darknet_model)?
    };

    Ok(())
}
