mod common;
mod config;
mod logging;
mod model;
mod train;
mod training_stream;
mod utils;

use crate::{common::*, config::Config, training_stream::TrainingStream};
use anyhow::{Context, Result};
use std::{path::PathBuf, sync::Arc};
use structopt::StructOpt;

#[derive(Debug, Clone, StructOpt)]
/// Train YOLO model
struct Args {
    #[structopt(long, default_value = "train.json5")]
    /// configuration file
    pub config_file: PathBuf,
}

#[tokio::main]
pub async fn main() -> Result<()> {
    // parse arguments
    let Args { config_file } = Args::from_args();
    let config = Arc::new(
        Config::open(&config_file)
            .with_context(|| format!("failed to load config file '{}'", config_file.display()))?,
    );

    let config = ArcRef::new(config);
    let logging_dir: Arc<Path> = {
        let start_time = Local::now();
        let dir = config
            .logging
            .dir
            .join(format!("{}", start_time.format(utils::FILE_STRFTIME)));
        dir.into_boxed_path().into()
    };
    let checkpoint_dir = Arc::new(logging_dir.join("checkpoints"));

    // create dirs and save config
    {
        tokio::fs::create_dir_all(&*logging_dir).await?;
        tokio::fs::create_dir_all(&*checkpoint_dir).await?;
        let path = logging_dir.join("config.json5");
        let text = serde_json::to_string_pretty(&*config)?;
        tokio::fs::write(&path, text).await?;
    }

    // create channels
    let (logging_tx, logging_rx) = broadcast::channel(2);
    let (data_tx, data_rx) = {
        let channel_size = match &config.training.device_config {
            config::DeviceConfig::SingleDevice { .. } => 2,
            config::DeviceConfig::MultiDevice { devices, .. } => devices.len() * 2,
            config::DeviceConfig::NonUniformMultiDevice { devices, .. } => devices.len() * 2,
        };
        tokio::sync::mpsc::channel(channel_size)
    };

    // load dataset
    info!("loading dataset");
    let dataset = TrainingStream::new(
        config.training.batch_size.get(),
        config.clone().map(|config| &config.dataset),
        config.clone().map(|config| &config.preprocessor),
        Some(logging_tx.clone()),
    )
    .await?;

    // start logger
    let logging_future = logging::logging_worker(config.clone(), logging_dir.clone(), logging_rx);

    // feeding worker
    let training_data_future = tokio::task::spawn(async move {
        let mut train_stream = dataset.train_stream()?;

        while let Some(result) = train_stream.next().await {
            let record = result?;
            data_tx
                .send(record)
                .await
                .map_err(|_| format_err!("failed to send message to training worker"))?;
        }

        anyhow::Ok(())
    })
    .map(|result| anyhow::Ok(result??));

    // training worker
    let training_worker_future = {
        let config = config.clone();
        let logging_tx = logging_tx.clone();
        let checkpoint_dir = checkpoint_dir.clone();

        tokio::task::spawn(async move {
            match config.training.device_config {
                config::DeviceConfig::SingleDevice { device } => {
                    tokio::task::spawn_blocking(move || {
                        train::single_gpu_training_worker(
                            config,
                            checkpoint_dir,
                            data_rx,
                            logging_tx,
                            device,
                        )
                    })
                    .await??;
                }
                config::DeviceConfig::MultiDevice { ref devices } => {
                    let batch_size = config.training.batch_size.get();
                    let minibatch_size = {
                        let num_devices = devices.len();
                        let div = batch_size / num_devices;
                        let rem = batch_size % num_devices;
                        (rem == 0).then(|| div).ok_or_else(|| {
                            format_err!("batch_size must be multiple of number of devices")
                        })?
                    };
                    info!("use minibatch size {} per device", minibatch_size);
                    let workers: Vec<_> = devices
                        .iter()
                        .cloned()
                        .map(|device| (device, minibatch_size))
                        .collect();

                    train::multi_gpu_training_worker(
                        config,
                        checkpoint_dir,
                        data_rx,
                        logging_tx,
                        &workers,
                    )
                    .await?;
                }
                config::DeviceConfig::NonUniformMultiDevice { ref devices } => {
                    let workers: Vec<_> = devices
                        .iter()
                        .map(|conf| {
                            let config::Worker {
                                minibatch_size,
                                device,
                            } = *conf;
                            (device, minibatch_size.get())
                        })
                        .collect();

                    train::multi_gpu_training_worker(
                        config,
                        checkpoint_dir,
                        data_rx,
                        logging_tx,
                        &workers,
                    )
                    .await?;
                }
            }

            anyhow::Ok(())
        })
        .map(|result| anyhow::Ok(result??))
    };

    futures::try_join!(training_data_future, training_worker_future, logging_future)?;

    Ok(())
}
