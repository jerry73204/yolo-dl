use anyhow::{format_err, Context, Result};
use darknet_config::{Darknet, DarknetModel, Graph, NodeKey};
use prettytable::{cell, row, Table};
use std::{
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
};
use structopt::StructOpt;

#[derive(Debug, Clone, StructOpt)]
/// Load darknet config and weights files and produce summary.
enum Args {
    Info {
        /// configuration file
        config_file: PathBuf,
    },
    TestWeightsFile {
        /// configuration file
        config_file: PathBuf,
        /// weights file
        weights_file: PathBuf,
    },
    MakeDotFile {
        /// configuration file
        config_file: PathBuf,
        /// output DOT file
        output_file: PathBuf,
    },
    DetectImage {
        /// configuration file
        config_file: PathBuf,
        /// weights file
        weights_file: PathBuf,
        /// image file
        image_file: PathBuf,
    },
}

fn main() -> Result<()> {
    pretty_env_logger::init();

    match Args::from_args() {
        Args::Info { config_file } => info(config_file)?,
        Args::TestWeightsFile {
            config_file,
            weights_file,
        } => test_weights_file(config_file, weights_file)?,
        Args::MakeDotFile {
            config_file,
            output_file,
        } => make_dot_file(config_file, output_file)?,
        Args::DetectImage {
            config_file,
            weights_file,
            image_file,
        } => {
            detect_image(config_file, weights_file, image_file)?;
        }
    }

    Ok(())
}

fn info(config_file: impl AsRef<Path>) -> Result<()> {
    let config = Darknet::load(config_file)?;
    let graph = Graph::from_config(&config)?;

    // print layer information
    {
        let mut table = Table::new();
        table.add_row(row![
            "index",
            "kind",
            "from indexes",
            "input shape",
            "output shape"
        ]);

        let num_layers = graph.layers.len() - 1;
        (0..num_layers).for_each(|index| {
            let layer = &graph.layers[&NodeKey::Index(index)];
            let kind = layer.as_ref();

            table.add_row(row![
                index,
                kind,
                layer.from_indexes(),
                layer.input_shape(),
                layer.output_shape()
            ]);
        });

        table.printstd();
    }

    Ok(())
}

fn test_weights_file(config_file: impl AsRef<Path>, weights_file: impl AsRef<Path>) -> Result<()> {
    let weights_file = weights_file.as_ref();
    let config = Darknet::load(config_file)?;
    let graph = Graph::from_config(&config)?;
    println!("loading weights file {} ...", weights_file.display());
    let mut darknet_model = DarknetModel::new(&graph)?;
    darknet_model
        .load_weights(weights_file)
        .with_context(|| "failed to load weights file")?;
    println!("weights file is loaded successfully!");
    Ok(())
}

#[cfg(feature = "dot")]
fn make_dot_file(config_file: impl AsRef<Path>, output_file: impl AsRef<Path>) -> Result<()> {
    let config = Darknet::load(config_file)?;
    let graph = Graph::from_config(&config)?;
    let mut writer = BufWriter::new(File::create(output_file)?);
    graph.render_dot(&mut writer)?;
    Ok(())
}

#[cfg(not(feature = "dot"))]
fn make_dot_file(_config_file: impl AsRef<Path>, _output_file: impl AsRef<Path>) -> Result<()> {
    use anyhow::bail;
    bail!("'dot' feature must be enabled to run this command");
}

#[cfg(feature = "tch")]
fn detect_image(
    config_file: impl AsRef<Path>,
    weights_file: impl AsRef<Path>,
    input_file: impl AsRef<Path>,
) -> Result<()> {
    use darknet_config::TchModel;
    use tch::{nn, vision, Device, Kind};
    use tch_goodies::TensorExt;

    // load model
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let root = vs.root();

    let mut model = {
        println!("loading config...");
        let mut darknet_model = DarknetModel::from_config_file(&config_file)?;

        println!("loading weights...");
        darknet_model.load_weights(&weights_file)?;

        println!("initializing model...");
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
    let image = vision::image::load(input_file)?;
    let image = image.resize2d_letterbox(in_h, in_w)?;
    let image = image.to_device(device).unsqueeze(0).to_kind(Kind::Float) / 255.0; // add batch dimension

    let _pred = model.forward_t(&image, false)?;

    println!("detection runs successfully");
    println!("detection visualization is not implemented");
    Ok(())
}

#[cfg(not(feature = "tch"))]
fn detect_image(
    config_file: impl AsRef<Path>,
    weights_file: impl AsRef<Path>,
    input_file: impl AsRef<Path>,
) -> Result<()> {
    use anyhow::bail;
    bail!("'tch' feature must be enabled to run this command");
}
