use anyhow::Result;
use argh::FromArgs;
use darknet_parse::{DarknetConfig, Layer, Model};
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
    let mut model = Model::from_config(&config)?;
    model.load_weights(weights_file)?;

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
                Layer::Convolutional(_) => "conv",
                Layer::Connected(_) => "connected",
                Layer::BatchNorm(_) => "batch_norm",
                Layer::Shortcut(_) => "shortcut",
                Layer::MaxPool(_) => "max_pool",
                Layer::Route(_) => "route",
                Layer::UpSample(_) => "up_sample",
                Layer::Yolo(_) => "yolo",
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

    Ok(())
}
