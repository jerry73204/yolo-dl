mod common;
mod config;
mod data;

use crate::{common::*, config::Config};

#[derive(Debug, Clone, FromArgs)]
/// Train YOLO model
struct Args {
    #[argh(option, default = "PathBuf::from(\"config.json5\")")]
    /// configuration file
    pub config_file: PathBuf,
}

#[async_std::main]
pub async fn main() -> Result<()> {
    let Args { config_file } = argh::from_env();
    let config = Arc::new(Config::open(&config_file)?);

    // load data set
    let (records, categories) = crate::data::train_stream(config.clone()).await?;
    let input_channels = 3;
    let num_classes = categories.len();

    // init model
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let model = yolo_dl::model::yolo_v5_small(&root, input_channels, num_classes);

    Ok(())
}