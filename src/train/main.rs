mod common;
mod config;
mod data;
mod logging;
mod message;
mod util;

use crate::{
    common::*,
    config::{Config, LoadCheckpoint},
    data::{DataSet, TrainingRecord},
    message::LoggingMessage,
    util::{RateCounter, Timing},
};

const CHECKPOINT_STRFTIME: &str = "%Y-%m-%d-%H-%M-%S-%3f";

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
    let session_id = Arc::new(Uuid::new_v4().to_string());

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

        if config.training.enable_multi_gpu {
            async_std::task::spawn(multi_gpu_training_worker(
                config,
                session_id,
                input_channels,
                num_classes,
                training_rx,
                logging_tx,
            ))
        } else {
            async_std::task::spawn_blocking(move || {
                training_worker(
                    config,
                    session_id,
                    input_channels,
                    num_classes,
                    training_rx,
                    logging_tx,
                )
            })
        }
    };

    futures::try_join!(training_data_future, training_worker_future, logging_future)?;

    Ok(())
}

async fn multi_gpu_training_worker(
    config: Arc<Config>,
    session_id: Arc<String>,
    input_channels: usize,
    num_classes: usize,
    data_rx: async_std::sync::Receiver<TrainingRecord>,
    logging_tx: broadcast::Sender<LoggingMessage>,
) -> Result<()> {
    struct WorkerContext {
        device: Device,
        minibatch_size: usize,
        vs: nn::VarStore,
        model: YoloModel,
        yolo_loss: YoloLoss,
        optimizer: nn::Optimizer<nn::Adam>,
    }

    let mut rate_counter = RateCounter::with_second_intertal();
    let master_device = config.training.master_device;
    let batch_size = config.training.batch_size.get();
    let default_minibatch_size = config.training.default_minibatch_size.get();
    let mut worker_contexts: Vec<_> = config
        .training
        .workers
        .iter()
        .map(|worker_config| -> Result<_> {
            let device = worker_config.device;
            let minibatch_size = worker_config
                .minibatch_size
                .map(|size| size.get())
                .unwrap_or(default_minibatch_size);
            let vs = nn::VarStore::new(device);
            let root = vs.root();
            let model = yolo_dl::model::yolo_v5_small(&root, input_channels, num_classes);
            let yolo_loss = YoloLossInit {
                reduction: Reduction::Sum,
                match_grid_method: Some(config.training.match_grid_method),
                iou_kind: Some(config.training.iou_kind),
                ..Default::default()
            }
            .build();
            let optimizer = nn::Adam::default().build(&vs, 0.01)?;

            Ok(WorkerContext {
                device,
                minibatch_size,
                vs,
                model,
                yolo_loss,
                optimizer,
            })
        })
        .try_collect()?;
    let mut training_step = 0;

    while let Ok(record) = data_rx.recv().await {
        let TrainingRecord {
            epoch,
            step: record_step,
            image,
            bboxes,
        } = record.to_device(master_device);

        // sync weights among workers
        {
            let mut iter = worker_contexts.into_iter();
            let first_context = iter.next().unwrap();
            let first_vs = Arc::new(first_context.vs);
            let other_contexts = future::try_join_all(iter.map(|mut context| {
                let first_vs = first_vs.clone();
                async_std::task::spawn_blocking(move || -> Result<_> {
                    context.vs.copy(&*first_vs)?;
                    Ok(context)
                })
            }))
            .await?;

            let first_context = WorkerContext {
                vs: Arc::try_unwrap(first_vs).unwrap(),
                ..first_context
            };

            worker_contexts = iter::once(first_context).chain(other_contexts).collect();
        }

        // forward step
        let outputs = {
            // distribute tasks to each worker
            let num_workers = worker_contexts.len();
            let mut batch_index = 0;
            let mut jobs = (0..num_workers).map(|_| vec![]).collect_vec();

            for (job_index, context) in worker_contexts.iter().cycle().enumerate() {
                let worker_index = job_index % num_workers;
                let batch_begin = batch_index;
                let batch_end = (batch_begin + context.minibatch_size).min(batch_size);
                let minibatch_size = batch_end - batch_begin;
                debug_assert!(minibatch_size > 0);

                let mini_image = image.narrow(0, batch_begin as i64, minibatch_size as i64);
                let mini_bboxes = bboxes[batch_begin..batch_end].to_owned();

                jobs[worker_index].push((job_index, mini_image, mini_bboxes));

                batch_index = batch_end;
                if batch_index == batch_size {
                    break;
                }
            }

            // run tasks
            let (worker_contexts_, outputs_per_worker) =
                future::join_all(worker_contexts.into_iter().zip_eq(jobs).map(
                    |(mut context, jobs)| {
                        async_std::task::spawn_blocking(move || {
                            let outputs = jobs
                                .into_iter()
                                .map(|(job_index, image, bboxes)| {
                                    let WorkerContext {
                                        device,
                                        ref vs,
                                        ref mut model,
                                        ref yolo_loss,
                                        ref mut optimizer,
                                        ..
                                    } = context;

                                    let image = image.to_device(device);

                                    // forward pass
                                    let output = model.forward_t(&image, true);

                                    // compute loss
                                    let losses = yolo_loss.forward(&output, &bboxes);

                                    // compute gradients
                                    let gradients = Tensor::run_backward(
                                        &[&losses.loss],
                                        &vs.trainable_variables(),
                                        false,
                                        false,
                                    );

                                    (job_index, output, losses, gradients)
                                })
                                .collect_vec();

                            (context, outputs)
                        })
                    },
                ))
                .await
                .into_iter()
                .unzip_n_vec();

            worker_contexts = worker_contexts_;

            let mut outputs = outputs_per_worker.into_iter().flatten().collect_vec();
            outputs.sort_by_cached_key(|(job_index, _output, _losses, _grad)| *job_index);
            outputs
        };

        // print message
        rate_counter.add(1.0);
        if let Some(batch_rate) = rate_counter.rate() {
            let record_rate = batch_rate * config.training.batch_size.get() as f64;
            info!(
                "epoch: {}\tstep: {}\t{:.2} mini-batches/s\t{:.2} records/s",
                epoch, training_step, batch_rate, record_rate
            );
        } else {
            info!("epoch: {}\tstep: {}", epoch, training_step);
        }

        training_step += 1;
    }
    Ok(())
}

fn training_worker(
    config: Arc<Config>,
    session_id: Arc<String>,
    input_channels: usize,
    num_classes: usize,
    training_rx: async_std::sync::Receiver<TrainingRecord>,
    logging_tx: broadcast::Sender<LoggingMessage>,
) -> Result<()> {
    warn!(r#"multi-gpu feature is disabled, use "master_device" as default device"#);

    // init model
    info!("initializing model");
    let device = config.training.master_device;
    let mut vs = nn::VarStore::new(device);
    let root = vs.root();
    let mut model = yolo_dl::model::yolo_v5_small(&root, input_channels, num_classes);
    let yolo_loss = YoloLossInit {
        match_grid_method: Some(config.training.match_grid_method),
        iou_kind: Some(config.training.iou_kind),
        ..Default::default()
    }
    .build();
    let mut optimizer = nn::Adam::default().build(&vs, 0.01)?;
    let mut rate_counter = RateCounter::with_second_intertal();
    let mut timing = Timing::new();
    let mut training_step = 0;
    let save_checkpoint_steps = config
        .training
        .save_checkpoint_steps
        .map(|steps| steps.get());
    let mut checkpoint_step_counter = 0;
    let checkpoint_dir = config.logging.dir.join("checkpoints");
    let saved_config_path = checkpoint_dir.join(format!("{}.json5", session_id));

    // create checkpoint dir
    std::fs::create_dir_all(&checkpoint_dir)?;

    // save config
    {
        let text = serde_json::to_string_pretty(&*config)?;
        std::fs::write(&saved_config_path, text)?;
    }

    // load checkpoint
    try_load_checkpoint(&mut vs, &checkpoint_dir, &config.training.load_checkpoint)?;

    // training
    info!("start training");

    while let Ok(record) = async_std::task::block_on(training_rx.recv()) {
        timing.set_record("next record");

        let TrainingRecord {
            epoch,
            step: record_step,
            image,
            bboxes,
        } = record.to_device(device);
        timing.set_record("to device");

        // forward pass
        let output = model.forward_t(&image, true);
        timing.set_record("forward");

        // compute loss
        let losses = yolo_loss.forward(&output, &bboxes);
        timing.set_record("loss");

        // optimizer
        optimizer.backward_step(&losses.loss);
        timing.set_record("backward");

        // print message
        rate_counter.add(1.0);
        if let Some(batch_rate) = rate_counter.rate() {
            let record_rate = batch_rate * config.training.batch_size.get() as f64;
            info!(
                "epoch: {}\tstep: {}\t{:.2} mini-batches/s\t{:.2} records/s",
                epoch, training_step, batch_rate, record_rate
            );
        } else {
            info!("epoch: {}\tstep: {}", epoch, training_step);
        }

        // save checkpoint
        if let Some(max_steps) = save_checkpoint_steps {
            checkpoint_step_counter += 1;
            if max_steps == checkpoint_step_counter {
                // reset counter
                checkpoint_step_counter = 0;

                // save model weights
                {
                    let filename = format!(
                        "{}_{}_{:06}_{:08.5}.ckpt",
                        session_id,
                        Local::now().format(CHECKPOINT_STRFTIME),
                        training_step,
                        f64::from(&losses.loss)
                    );
                    let path = checkpoint_dir.join(filename);
                    vs.save(&path)?;
                }
            }
        }

        // send to logger
        {
            let msg = LoggingMessage::new_training_step("loss", training_step, &losses);
            logging_tx
                .send(msg)
                .map_err(|_err| format_err!("cannot send message to logger"))?;
        }
        {
            let msg = LoggingMessage::new_training_output(
                "output",
                training_step,
                &image,
                &output,
                &losses,
            );
            logging_tx
                .send(msg)
                .map_err(|_err| format_err!("cannot send message to logger"))?;
        }

        // info!("{:#?}", timing.records());
        timing = Timing::new();
        training_step += 1;
    }

    Fallible::Ok(())
}

fn try_load_checkpoint(
    vs: &mut nn::VarStore,
    checkpoint_dir: &Path,
    load_checkpoint: &LoadCheckpoint,
) -> Result<()> {
    let checkpoint_filename_regex =
        Regex::new(r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{3}_(\d{6})_\d+.\d+.ckpt$").unwrap();

    let path = match load_checkpoint {
        LoadCheckpoint::Disabled => {
            info!("checkpoint loading is disabled");
            None
        }
        LoadCheckpoint::FromRecent => {
            let path_step = checkpoint_dir
                .read_dir()?
                .map(|result| -> Result<_> {
                    let entry = result?;
                    if entry.file_type()?.is_file() {
                        if let Ok(file_name) = entry.file_name().into_string() {
                            match checkpoint_filename_regex.captures(&file_name) {
                                Some(captures) => {
                                    let step: usize =
                                        captures.get(2).unwrap().as_str().parse().unwrap();
                                    let path = entry.path();
                                    Ok(Some((path, step)))
                                }
                                None => Ok(None),
                            }
                        } else {
                            Ok(None)
                        }
                    } else {
                        Ok(None)
                    }
                })
                .filter_map(|result| result.transpose())
                .fold(Ok(None), |prev_result, curr_result| -> Result<_> {
                    let prev = prev_result?;
                    let curr = curr_result?;
                    match prev {
                        Some(prev) => {
                            let (_prev_path, prev_step) = &prev;
                            let (_curr_path, curr_step) = &curr;
                            if curr_step > prev_step {
                                Ok(Some(curr))
                            } else {
                                Ok(Some(prev))
                            }
                        }
                        None => Ok(Some(curr)),
                    }
                })?;
            let path = path_step.map(|(path, _step)| path);

            if let None = &path {
                warn!("no recent checkpoint files found");
            }

            path
        }
        LoadCheckpoint::FromSession { session_id } => {
            let path_step = checkpoint_dir
                .read_dir()?
                .map(|result| -> Result<_> {
                    let entry = result?;
                    if entry.file_type()?.is_file() {
                        if let Ok(file_name) = entry.file_name().into_string() {
                            match checkpoint_filename_regex.captures(&file_name) {
                                Some(captures) => {
                                    let uuid = captures.get(1).unwrap().as_str();
                                    if uuid == session_id {
                                        let step: usize =
                                            captures.get(2).unwrap().as_str().parse().unwrap();
                                        let path = entry.path();
                                        Ok(Some((path, step)))
                                    } else {
                                        Ok(None)
                                    }
                                }
                                None => Ok(None),
                            }
                        } else {
                            Ok(None)
                        }
                    } else {
                        Ok(None)
                    }
                })
                .filter_map(|result| result.transpose())
                .fold(Ok(None), |prev_result, curr_result| -> Result<_> {
                    let prev = prev_result?;
                    let curr = curr_result?;
                    match prev {
                        Some(prev) => {
                            let (_prev_path, prev_step) = &prev;
                            let (_curr_path, curr_step) = &curr;
                            if curr_step > prev_step {
                                Ok(Some(curr))
                            } else {
                                Ok(Some(prev))
                            }
                        }
                        None => Ok(Some(curr)),
                    }
                })?;
            let path = path_step.map(|(path, _step)| path);

            if let None = &path {
                warn!(
                    "no recent checkpoint files found for session {}",
                    session_id
                );
            }

            path
        }
        LoadCheckpoint::FromFile { file } => {
            if file.is_file() {
                Some(file.to_owned())
            } else {
                warn!("{} is not a file", file.display());
                None
            }
        }
    };

    if let Some(path) = path {
        info!("load checkpoint file {}", path.display());
        vs.load_partial(path)?;
    }

    Ok(())
}
