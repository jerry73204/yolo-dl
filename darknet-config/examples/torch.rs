use anyhow::{format_err, Context, Result};
use argh::FromArgs;
use darknet_config::{DarknetModel, TchModel};
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

    // load model
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let root = vs.root();

    let mut model = {
        info!("loading config...");
        let mut darknet_model = DarknetModel::from_config_file(&config_file)?;

        info!("loading weights...");
        darknet_model.load_weights(&weights_file)?;

        info!("initializing model...");
        TchModel::from_darknet_model(&root, &darknet_model)?
    };

    let [_in_c, in_h, in_w] = {
        let [in_h, in_w, in_c] = model
            .input_shape()
            .single_hwc()
            .ok_or_else(|| format_err!("only model with 3D input is supported"))?;
        [in_c as i64, in_h as i64, in_w as i64]
    };

    // load image
    let image = vision::image::load(image_file)?;
    let image = image.resize2d_letterbox(in_h, in_w)?;
    let image = image.to_device(device).unsqueeze(0).to_kind(Kind::Float) / 255.0; // add batch dimension

    let pred = model.forward_t(&image, false)?;

    Ok(())
}
