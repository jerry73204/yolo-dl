use anyhow::{format_err, Context, Result};
use darknet_config::{DarknetModel, TchModel};
use darknet_test::{
    config::{Config, InputConfig},
    darknet::network::Network,
};
use itertools::Itertools;
use log::info;
use ndarray::{Array3, ArrayD};
use std::{convert::TryInto, fs, path::PathBuf};
pub use structopt::StructOpt;
use tch::{nn, vision, Kind};
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
    let mut darknet_model = Network::load(&config_file, Some(&weights_file), false)?;

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

            // forward rust model
            let rust_input = vision::image::load(&image_file)?
                .resize2d_letterbox(in_h as i64, in_w as i64)?
                .to_device(rust_device)
                .to_kind(Kind::Float)
                .g_div1(255.0)
                .view([1, in_c as i64, in_h as i64, in_w as i64]);

            let _rust_output = rust_model.forward_t(&rust_input, false)?;

            // forward darknet model
            let darknet_input: Array3<f32> = {
                let array: ArrayD<f32> = (&rust_input
                    .view([in_c as i64, in_h as i64, in_w as i64])
                    .permute(&[2, 1, 0]))
                    .try_into()?;
                array.into_shape((in_w, in_w, in_c)).unwrap()
            };

            let _darknet_output = darknet_model.predict(&darknet_input, 0.8, 0.5, 0.45, true);

            Ok(())
        })?;

    Ok(())
}
