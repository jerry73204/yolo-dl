mod common;
mod config;
mod data;
mod logging;
mod message;
mod util;

use crate::{
    common::*,
    config::Config,
    data::{DataSet, TrainingRecord},
    message::LoggingMessage,
    util::{RateCounter, Timing},
};

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
    let training_data_future = {
        let logging_tx = logging_tx.clone();
        async_std::task::spawn(async move {
            let mut train_stream = dataset.train_stream(logging_tx).await?;

            while let Some(result) = train_stream.next().await {
                let record = result?;
                training_tx.send(record).await;
            }

            Fallible::Ok(())
        })
    };

    // training worker
    let training_worker_future = {
        let config = config.clone();
        let logging_tx = logging_tx.clone();

        async_std::task::spawn_blocking(move || {
            train_worker(config, input_channels, num_classes, training_rx, logging_tx)
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
    logging_tx: broadcast::Sender<LoggingMessage>,
) -> Fallible<()> {
    // init model
    info!("initializing model");
    let vs = nn::VarStore::new(config.device);
    let root = vs.root();
    let model = yolo_dl::model::yolo_v5_small(&root, input_channels, num_classes);
    let yolo_loss = YoloLossInit {
        match_grid_method: Some(config.match_grid_method),
        iou_kind: Some(config.iou_kind),
        ..Default::default()
    }
    .build();
    let mut optimizer = nn::Adam::default().build(&vs, 0.01)?;
    let mut rate_counter = RateCounter::with_second_intertal();
    let mut timing = Timing::new();

    // training
    info!("start training");

    while let Ok(record) = async_std::task::block_on(training_rx.recv()) {
        timing.set_record("next record");

        let TrainingRecord {
            epoch,
            step,
            image,
            bboxes,
        } = record.to_device(config.device);
        timing.set_record("to device");

        // forward pass
        let output = model.forward_t(&image, true);
        timing.set_record("forward");

        // compute loss
        let loss = yolo_loss.forward(&output, &bboxes);
        timing.set_record("loss");

        // optimizer
        optimizer.backward_step(&loss.loss);
        timing.set_record("backward");

        // print message
        rate_counter.add(1.0);
        if let Some(batch_rate) = rate_counter.rate() {
            let record_rate = batch_rate * config.mini_batch_size as f64;
            info!(
                "epoch: {}\tstep: {}\t{:.2} mini-batches/s\t{:.2} records/s",
                epoch, step, batch_rate, record_rate
            );
        } else {
            info!("epoch: {}\tstep: {}", epoch, step);
        }

        // send to logger
        {
            let msg = LoggingMessage::new_training_step("loss", step, (&loss.loss).into());
            logging_tx
                .send(msg)
                .map_err(|_err| format_err!("cannot send message to logger"))?;
        }
        {
            let msg = LoggingMessage::new_training_output("output", &image, &output, &loss);
            logging_tx
                .send(msg)
                .map_err(|_err| format_err!("cannot send message to logger"))?;
        }

        // info!("{:#?}", timing.records());
        timing = Timing::new();
    }

    Fallible::Ok(())
}
