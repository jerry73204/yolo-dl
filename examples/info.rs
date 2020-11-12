use anyhow::{Context, Result};
use argh::FromArgs;
use darknet_config::{DarknetConfig, DarknetModel, LayerBase, ModelBase};
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
    let model = ModelBase::from_config(&config)?;

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

        let num_layers = model.layers.len();
        (0..num_layers).for_each(|index| {
            let layer = &model.layers[&index];

            let kind = match layer {
                LayerBase::Convolutional(_) => "conv",
                LayerBase::Connected(_) => "connected",
                LayerBase::BatchNorm(_) => "batch_norm",
                LayerBase::Shortcut(_) => "shortcut",
                LayerBase::MaxPool(_) => "max_pool",
                LayerBase::Route(_) => "route",
                LayerBase::UpSample(_) => "up_sample",
                LayerBase::Yolo(_) => "yolo",
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
    let mut darknet_model = DarknetModel::new(&model)?;
    darknet_model
        .load_weights(weights_file)
        .with_context(|| "failed to load weights file")?;
    println!("weights file is successfully loaded!");

    Ok(())
}
