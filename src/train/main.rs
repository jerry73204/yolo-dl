mod common;
mod config;
mod data;
mod logging;
mod message;
mod util;

use crate::{common::*, config::Config, data::DataSet, data::TrainingRecord, util::RateCounter};

#[derive(Debug, Clone, FromArgs)]
/// Train YOLO model
struct Args {
    #[argh(option, default = "PathBuf::from(\"train.json5\")")]
    /// configuration file
    pub config_file: PathBuf,
}

#[async_std::main]
pub async fn main() -> Result<()> {
    pretty_env_logger::init();

    let Args { config_file } = argh::from_env();
    let config = Arc::new(Config::open(&config_file)?);

    // load dataset
    info!("loading dataset");
    let dataset = DataSet::new(config.clone()).await?;
    let input_channels = dataset.input_channels();
    let num_classes = dataset.num_classes();

    // create channels
    let (logging_tx, logging_rx) = broadcast::channel(2);
    let (training_tx, training_rx) = async_std::sync::channel(2);

    // start logger
    let logging_future = logging::logging_worker(config.clone(), logging_rx).await?;

    // feeding worker
    let training_data_future = async_std::task::spawn(async move {
        let mut train_stream = dataset.train_stream(logging_tx.clone()).await?;

        while let Some(result) = train_stream.next().await {
            let record = result?;
            training_tx.send(record).await;
        }

        Fallible::Ok(())
    });

    let training_worker_future = {
        let config = config.clone();

        async_std::task::spawn_blocking(move || {
            train_worker(config, input_channels, num_classes, training_rx)
        })
    };

    futures::try_join!(training_data_future, training_worker_future, logging_future)?;

    Ok(())
}

fn train_worker(
    config: Arc<Config>,
    input_channels: usize,
    num_classes: usize,
    training_rx: async_std::sync::Receiver<TrainingRecord>,
) -> Fallible<()> {
    // init model
    info!("initializing model");
    let vs = nn::VarStore::new(config.device);
    let root = vs.root();
    let model = yolo_dl::model::yolo_v5_small(&root, input_channels, num_classes);
    let yolo_loss = YoloLossInit::default().build();
    let mut optimizer = nn::Adam::default().build(&vs, 0.01)?;
    let mut rate = 0.0;

    // training
    info!("start training");
    // let mut rate_counter = RateCounter::new(0.9);

    while let Ok(record) = async_std::task::block_on(training_rx.recv()) {
        let TrainingRecord {
            epoch,
            step,
            image,
            bboxes,
        } = record.to_device(config.device);

        // forward pass
        let output = model.forward_t(&image, true);

        // compute loss
        let loss = yolo_loss.forward(&output, &bboxes);

        // optimizer
        optimizer.backward_step(&loss);

        // print message
        // rate_counter.add(1.0).await;
        // rate = rate_counter.rate().await.unwrap_or(rate);
        info!("epoch: {}\tstep: {}", epoch, step);
    }

    Fallible::Ok(())
}