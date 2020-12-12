use anyhow::{format_err, Context, Result};
use argh::FromArgs;
use darknet_config::{DarknetModel, TchModel};
use darknet_test::{network::Network, sys};
use log::info;
use std::path::PathBuf;
use tch::{nn, vision, Device, Kind};
use tch_goodies::TensorExt;

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

    // load rust model
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let root = vs.root();

    let mut rust_model = {
        info!("loading config...");
        let mut darknet_model = DarknetModel::from_config_file(&config_file)?;

        info!("loading weights...");
        darknet_model.load_weights(&weights_file)?;

        info!("initializing model...");
        TchModel::from_darknet_model(&root, &darknet_model)?
    };

    // load darknet model
    {
        let network_ptr = Network::load(&config_file, Some(weights_file), false)?.into_raw();
    }

    Ok(())
}
