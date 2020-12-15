use anyhow::{ensure, Result};
use approx::abs_diff_eq;
use darknet_config::{torch::LayerOutput, DarknetModel, TchModel};
use darknet_test::{
    config::{Config, InputConfig},
    darknet::network::Network,
};
use itertools::Itertools;
use log::info;
use ndarray::{Array3, ArrayD};
use std::{
    convert::{TryFrom, TryInto},
    fs,
    path::PathBuf,
};
pub use structopt::StructOpt;
use tch::{nn, vision, Kind, Tensor};
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
        .enumerate()
        .try_for_each(|(_step, image_file)| -> Result<_> {
            info!("test image {}", image_file.display());

            // forward rust model
            let rust_input = vision::image::load(&image_file)?
                .resize2d_letterbox(in_h as i64, in_w as i64)?
                .to_device(rust_device)
                .to_kind(Kind::Float)
                .g_div1(255.0)
                .view([1, in_c as i64, in_h as i64, in_w as i64]);

            let (_rust_output, rust_feature_maps) = rust_model.forward_t(&rust_input, false)?;

            // forward darknet model
            let darknet_input: Array3<f32> = {
                let array: ArrayD<f32> = (&rust_input
                    .view([in_c as i64, in_h as i64, in_w as i64])
                    .permute(&[2, 1, 0]))
                    .try_into()?;
                array.into_shape((in_w, in_w, in_c)).unwrap()
            };

            let _darknet_output = darknet_model.predict(&darknet_input, 0.8, 0.5, 0.45, true);

            // verify per-layer output
            darknet_model
                .layers()
                .iter()
                .zip_eq(rust_feature_maps.into_iter())
                .enumerate()
                .try_for_each(
                    |(layer_index, (darknet_layer, rust_feature_map))| -> Result<_> {
                        // dbg!(darknet_layer.type_());

                        let darknet_feature_map = darknet_layer.output_array();

                        match rust_feature_map {
                            LayerOutput::Tensor(rust_feature_map) => {
                                // check shape
                                {
                                    let rust_shape = rust_feature_map.size();
                                    let dark_shape = darknet_feature_map.shape();

                                    let is_shape_correct = match (rust_shape.as_slice(), dark_shape)
                                    {
                                        (
                                            &[rust_b, rust_c, rust_h, rust_w],
                                            &[dark_b, dark_w, dark_h, dark_c],
                                        ) => {
                                            rust_b as usize == dark_b
                                                && rust_c as usize == dark_c
                                                && rust_h as usize == dark_h
                                                && rust_w as usize == dark_w
                                        }
                                        (&[rust_b, rust_c], &[dark_b, dark_c]) => {
                                            rust_b as usize == dark_b && rust_c as usize == dark_c
                                        }
                                        _ => false,
                                    };

                                    ensure!(
                                        is_shape_correct,
                                        "shape mismatch at layer {}: rust_shape={:?}, darknet_shape={:?}",
                                        layer_index,
                                        rust_shape,
                                        dark_shape
                                    );
                                }

                                // convert feature map type
                                let darknet_feature_map =
                                    Tensor::try_from(darknet_feature_map.into_owned())?
                                        // [b, w, h, c] to [b, c, h, w]
                                        .permute(&[0, 3, 2, 1])
                                        .to_device(rust_device);

                                // {
                                //     let num_buckets = 10;
                                //     let darknet_hist =
                                //         Vec::<f32>::from(&darknet_feature_map.histc(num_buckets));
                                //     let rust_hist =
                                //         Vec::<f32>::from(&rust_feature_map.histc(num_buckets));
                                //     dbg!(darknet_hist, rust_hist);
                                // }

                                // check values
                                {
                                    let mse = f32::from(
                                        (rust_feature_map - darknet_feature_map)
                                            .pow(2.0)
                                            .mean(Kind::Float),
                                    );

                                    // check mse
                                    ensure!(
                                        abs_diff_eq!(mse, 0.0),
                                        "output differs at layer {}. mse={}",
                                        layer_index,
                                        mse
                                    );
                                }
                            }
                            LayerOutput::Yolo(_rust_feature_map) => {
                                // TODO
                            }
                        }

                        Ok(())
                    },
                )?;

            Ok(())
        })?;

    Ok(())
}
