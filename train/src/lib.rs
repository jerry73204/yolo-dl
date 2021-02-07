mod common;
pub mod config;
mod data;
mod logging;
mod message;
mod model;
mod train;
mod utils;

use crate::{
    common::*,
    config::{Config, DeviceConfig, WorkerConfig},
    data::{GenericDataset, TrainingStream},
};

pub async fn start(config: Arc<Config>) -> Result<()> {
    let start_time = Local::now();
    let logging_dir = Arc::new(
        config
            .logging
            .dir
            .join(format!("{}", start_time.format(utils::FILE_STRFTIME))),
    );
    let checkpoint_dir = Arc::new(logging_dir.join("checkpoints"));

    // create dirs and save config
    {
        tokio::fs::create_dir_all(&*logging_dir).await?;
        tokio::fs::create_dir_all(&*checkpoint_dir).await?;
        let path = logging_dir.join("config.json5");
        let text = serde_json::to_string_pretty(&*config)?;
        std::fs::write(&path, text)?;
    }

    // create channels
    let (logging_tx, logging_rx) = broadcast::channel(2);
    let (data_tx, data_rx) = {
        let channel_size = match &config.training.device_config {
            DeviceConfig::SingleDevice { .. } => 2,
            DeviceConfig::MultiDevice { devices, .. } => devices.len() * 2,
            DeviceConfig::NonUniformMultiDevice { devices, .. } => devices.len() * 2,
        };
        async_std::channel::bounded(channel_size)
    };

    // load dataset
    info!("loading dataset");
    let dataset = TrainingStream::new(config.clone(), logging_tx.clone()).await?;
    let input_channels = dataset.input_channels();
    let num_classes = dataset.classes().len();

    // start logger
    let logging_future =
        logging::logging_worker(config.clone(), logging_dir.clone(), logging_rx).await?;

    // feeding worker
    let training_data_future = tokio::task::spawn(async move {
        let mut train_stream = dataset.train_stream().await?;

        while let Some(result) = train_stream.next().await {
            let record = result?;
            data_tx
                .send(record)
                .await
                .map_err(|_| format_err!("failed to send message to training worker"))?;
        }

        Fallible::Ok(())
    })
    .map(|result| Fallible::Ok(result??));

    // training worker
    let training_worker_future = {
        let config = config.clone();
        let logging_tx = logging_tx.clone();
        let logging_dir = logging_dir.clone();
        let checkpoint_dir = checkpoint_dir.clone();

        async move {
            match config.training.device_config {
                DeviceConfig::SingleDevice { device } => {
                    tokio::task::spawn_blocking(move || {
                        train::single_gpu_training_worker(
                            config,
                            logging_dir,
                            checkpoint_dir,
                            input_channels,
                            num_classes,
                            data_rx,
                            logging_tx,
                            device,
                        )
                    })
                    .map(|result| Fallible::Ok(result??))
                    .await?;
                }
                DeviceConfig::MultiDevice {
                    minibatch_size,
                    ref devices,
                } => {
                    let minibatch_size = minibatch_size.get();
                    let workers: Vec<_> = devices
                        .iter()
                        .cloned()
                        .map(|device| (device, minibatch_size))
                        .collect();

                    tokio::task::spawn(async move {
                        train::multi_gpu_training_worker(
                            config,
                            logging_dir,
                            checkpoint_dir,
                            input_channels,
                            num_classes,
                            data_rx,
                            logging_tx,
                            &workers,
                        )
                        .await
                    })
                    .map(|result| Fallible::Ok(result??))
                    .await?;
                }
                DeviceConfig::NonUniformMultiDevice { ref devices } => {
                    let workers: Vec<_> = devices
                        .iter()
                        .map(|conf| {
                            let WorkerConfig {
                                minibatch_size,
                                device,
                            } = *conf;
                            (device, minibatch_size.get())
                        })
                        .collect();

                    tokio::task::spawn(async move {
                        train::multi_gpu_training_worker(
                            config,
                            logging_dir,
                            checkpoint_dir,
                            input_channels,
                            num_classes,
                            data_rx,
                            logging_tx,
                            &workers,
                        )
                        .await
                    })
                    .map(|result| Fallible::Ok(result??))
                    .await?;
                }
            }

            Fallible::Ok(())
        }
    };

    futures::try_join!(training_data_future, training_worker_future, logging_future)?;

    Ok(())
}
