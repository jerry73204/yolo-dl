mod common;
mod config;
mod data;
mod logging;
mod message;
mod unit;
mod util;

use crate::{common::*, config::Config, data::DataSet, util::RateCounter};

#[derive(Debug, Clone, FromArgs)]
/// Train YOLO model
struct Args {
    #[argh(option, default = "PathBuf::from(\"config.json5\")")]
    /// configuration file
    pub config_file: PathBuf,
}

#[async_std::main]
pub async fn main() -> Result<()> {
    pretty_env_logger::init();

    let Args { config_file } = argh::from_env();
    let config = Arc::new(Config::open(&config_file)?);

    // create channels
    let (logging_tx, logging_rx) = broadcast::channel(2);

    // start logger
    let logging_future = logging::logging_worker(config.clone(), logging_rx).await?;

    // load data set
    info!("loading dataset");
    let dataset = DataSet::new(config.clone()).await?;
    let input_channels = dataset.input_channels();
    let num_classes = dataset.num_classes();

    let mut train_stream = dataset.train_stream(logging_tx.clone()).await?;

    // init model
    info!("initializing model");
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let model = yolo_dl::model::yolo_v5_small(&root, input_channels, num_classes);

    // training
    info!("start training");
    let mut rate_counter = RateCounter::new(0.9);

    while let Some(result) = train_stream.next().await {
        let record = result?;

        rate_counter.add(1.0).await;
        if let Some(rate) = rate_counter.rate().await {
            info!("rate {:.2} msg/s", rate);
        }
    }

    Ok(())
}
