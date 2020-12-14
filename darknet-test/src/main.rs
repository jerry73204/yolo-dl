use anyhow::{format_err, Context, Result};
use darknet_config::{DarknetModel, TchModel};
use darknet_test::{
    config::{Config, InputConfig},
    darknet::network::Network,
    sys,
};
use itertools::Itertools;
use log::info;
use ndarray::ArrayD;
use std::{convert::TryInto, fs, path::PathBuf};
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
        input,
    } = json5::from_str(&fs::read_to_string(&darknet_test_config_file)?)?;

    // load darknet model
    info!("loading darknet model ...");
    let darknet_model = Network::load(&config_file, Some(&weights_file), false)?;

    // load rust model
    info!("loading rust model ...");
    let vs = nn::VarStore::new(rust_device);
    let root = vs.root();

    let mut rust_model = {
        let mut darknet_model = DarknetModel::from_config_file(&config_file)?;
        darknet_model.load_weights(&weights_file)?;
        TchModel::from_darknet_model(&root, &darknet_model)?
    };

    // get input shape
    let (in_c, in_h, in_w) = darknet_model.input_shape();
    info!(
        "input shape channels={} height={} width={}",
        in_c, in_h, in_w
    );

    // find image files
    info!("scan image files ...");
    let image_files: Vec<_> = match input {
        InputConfig::File(file) => vec![file],
        InputConfig::Dir(dir) => dir
            .read_dir()?
            .map(|result| -> Result<_> {
                let entry = result?;
                if entry.metadata()?.is_file() {
                    Ok(Some(entry.path()))
                } else {
                    Ok(None)
                }
            })
            .filter_map(|result| result.transpose())
            .try_collect()?,
        InputConfig::Glob(pattern) => glob::glob(&pattern)?
            .map(|result| -> Result<_> {
                let path = result?;
                if path.metadata()?.is_file() {
                    Ok(Some(path))
                } else {
                    Ok(None)
                }
            })
            .filter_map(|result| result.transpose())
            .try_collect()?,
    };

    image_files
        .into_iter()
        .try_for_each(|image_file| -> Result<_> {
            info!("test image {}", image_file.display());

            let rust_input = vision::image::load(&image_file)?
                .resize2d_letterbox(in_h as i64, in_w as i64)?
                .to_device(rust_device)
                .to_kind(Kind::Float)
                .g_div1(255.0)
                .view([1, in_c as i64, in_h as i64, in_w as i64]);

            let rust_output = rust_model.forward_t(&rust_input, false)?;

            // let darknet_input: ArrayD<f32> = (&rust_input).try_into()?;

            Ok(())
        })?;

    Ok(())
}
