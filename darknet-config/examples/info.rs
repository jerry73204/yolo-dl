use anyhow::{Context, Result};
use argh::FromArgs;
use darknet_config::{DarknetConfig, DarknetModel, Graph, Node};
use prettytable::{cell, row, Table};
use std::path::PathBuf;

#[derive(Debug, Clone, FromArgs)]
/// Load darknet config and weights files and produce summary.
struct Args {
    #[argh(positional)]
    /// configuration file
    config_file: PathBuf,
    #[argh(positional)]
    /// weights file
    weights_file: PathBuf,
}

fn main() -> Result<()> {
    pretty_env_logger::init();

    let Args {
        config_file,
        weights_file,
    } = argh::from_env();

    let config = DarknetConfig::load(config_file)?;
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

        let num_layers = graph.layers.len();
        (0..num_layers).for_each(|index| {
            let layer = &graph.layers[&index];

            let kind = match layer {
                Node::Convolutional(_) => "conv",
                Node::Connected(_) => "connected",
                Node::BatchNorm(_) => "batch_norm",
                Node::Dropout(_) => "dropout",
                Node::Softmax(_) => "softmax",
                Node::Shortcut(_) => "shortcut",
                Node::MaxPool(_) => "max_pool",
                Node::Route(_) => "route",
                Node::UpSample(_) => "up_sample",
                Node::Yolo(_) => "yolo",
                Node::GaussianYolo(_) => "gaussian_yolo",
            };

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

    println!("loading weights file {}", weights_file.display());
    let mut darknet_model = DarknetModel::new(&graph)?;
    darknet_model
        .load_weights(weights_file)
        .with_context(|| "failed to load weights file")?;
    println!("weights file is successfully loaded!");

    Ok(())
}
