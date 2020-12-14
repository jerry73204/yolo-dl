use anyhow::{format_err, Context, Result};
use argh::FromArgs;
use darknet_config::{DarknetModel, TchModel};
use darknet_test::{config::Config, darknet::network::Network, sys};
use log::info;
use std::{fs, path::PathBuf};
pub use structopt::StructOpt;
use tch::{nn, vision, Device, Kind};
use tch_goodies::TensorExt;

#[derive(Debug, Clone, StructOpt)]
/// Load darknet config and weights files and produce summary.
struct Args {
    #[structopt(long, default_value = "darknet-test.json5")]
    config_file: PathBuf,
}

fn main() -> Result<()> {
    pretty_env_logger::init();

    let Args {
        config_file: darknet_test_config_file,
    } = Args::from_args();
    let Config {
        config_file,
        weights_file,
        rust_device,
    } = json5::from_str(&fs::read_to_string(&darknet_test_config_file)?)?;

    // load darknet model
    {
        info!("loading darknet model");
        let network_ptr = Network::load(&config_file, Some(&weights_file), false)?.into_raw();
    }

    // load rust model
    let vs = nn::VarStore::new(rust_device);
    let root = vs.root();

    let mut rust_model = {
        info!("loading config...");
        let mut darknet_model = DarknetModel::from_config_file(&config_file)?;

        info!("loading weights...");
        darknet_model.load_weights(&weights_file)?;

        info!("initializing model...");
        TchModel::from_darknet_model(&root, &darknet_model)?
    };

    Ok(())
}
