mod common;
mod config;
mod data;
mod logging;
mod message;
mod util;

use crate::{
    common::*,
    config::{
        Config, DeviceConfig, LoadCheckpoint, LoggingConfig, LossConfig, TrainingConfig,
        WorkerConfig,
    },
    data::{GenericDataset, TrainingRecord, TrainingStream},
    message::LoggingMessage,
    util::{LrScheduler, RateCounter},
};

const FILE_STRFTIME: &str = "%Y-%m-%d-%H-%M-%S.%3f%z";

#[derive(Debug, Clone, StructOpt)]
/// Train YOLO model
struct Args {
    #[structopt(long, default_value = "train.json5")]
    /// configuration file
    pub config_file: PathBuf,
}

#[tracing::instrument]
#[async_std::main]
pub async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let Args { config_file } = Args::from_args();
    let config = Arc::new(
        Config::open(&config_file)
            .with_context(|| format!("failed to load config file '{}'", config_file.display()))?,
    );
    let start_time = Local::now();
    let logging_dir = Arc::new(
        config
            .logging
            .dir
            .join(format!("{}", start_time.format(FILE_STRFTIME))),
    );
    let checkpoint_dir = Arc::new(logging_dir.join("checkpoints"));

    // create dirs and save config
    {
        async_std::fs::create_dir_all(&*logging_dir).await?;
        async_std::fs::create_dir_all(&*checkpoint_dir).await?;
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
    let training_data_future = {
        async_std::task::spawn(async move {
            let mut train_stream = dataset.train_stream().await?;

            while let Some(result) = train_stream.next().await {
                let record = result?;
                data_tx.send(record).await;
            }

            Fallible::Ok(())
        })
    };

    // training worker
    let training_worker_future = {
        let config = config.clone();
        let logging_tx = logging_tx.clone();
        let logging_dir = logging_dir.clone();
        let checkpoint_dir = checkpoint_dir.clone();

        async move {
            match config.training.device_config {
                DeviceConfig::SingleDevice { device } => {
                    async_std::task::spawn_blocking(move || {
                        single_gpu_training_worker(
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

                    async_std::task::spawn(async move {
                        multi_gpu_training_worker(
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

                    async_std::task::spawn(async move {
                        multi_gpu_training_worker(
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
                    .await?;
                }
            }

            Fallible::Ok(())
        }
    };

    futures::try_join!(training_data_future, training_worker_future, logging_future)?;

    Ok(())
}

async fn multi_gpu_training_worker(
    config: Arc<Config>,
    _logging_dir: Arc<PathBuf>,
    checkpoint_dir: Arc<PathBuf>,
    input_channels: usize,
    num_classes: usize,
    data_rx: async_std::channel::Receiver<TrainingRecord>,
    logging_tx: broadcast::Sender<LoggingMessage>,
    workers: &[(Device, usize)],
) -> Result<()> {
    // types

    struct WorkerContext {
        device: Device,
        minibatch_size: usize,
        vs: nn::VarStore,
        model: YoloModel,
        yolo_loss: YoloLoss,
        optimizer: nn::Optimizer<nn::Adam>,
    }

    struct WorkerJob {
        job_index: usize,
        minibatch_size: usize,
        image: Tensor,
        bboxes: Vec<Vec<LabeledRatioBBox>>,
    }

    struct WorkerOutput {
        job_index: usize,
        worker_index: usize,
        minibatch_size: usize,
        output: YoloOutput,
        losses: YoloLossOutput,
        target_bboxes: HashMap<Arc<InstanceIndex>, Arc<LabeledGridBBox<R64>>>,
        gradients: Vec<Tensor>,
    }

    // initialization
    info!(
        "use device configuration (device, minibatch_size): {:?}",
        workers
    );

    let Config {
        training:
            TrainingConfig {
                initial_step: init_training_step,
                ref lr_schedule,
                batch_size,
                save_checkpoint_steps,
                ..
            },
        logging: LoggingConfig {
            enable_training_output,
            ..
        },
        ..
    } = *config;
    let (master_device, _) = *workers
        .get(0)
        .ok_or_else(|| format_err!("workers list cannot be empty"))?;
    let mut lr_scheduler = LrScheduler::new(lr_schedule, init_training_step)?;
    let mut training_step = init_training_step;
    let mut rate_counter = RateCounter::with_second_intertal();
    let batch_size = batch_size.get();
    let save_checkpoint_steps = save_checkpoint_steps.map(|steps| steps.get());
    let mut init_timing = Timing::new("initialization");

    if let None = save_checkpoint_steps {
        warn!("checkpoint saving is disabled");
    }

    // initialize workers
    info!("initializing model");

    let mut worker_contexts = {
        let init_lr = lr_scheduler.next();

        future::try_join_all(workers.iter().cloned().map(|(device, minibatch_size)| {
            let TrainingConfig {
                loss:
                    LossConfig {
                        match_grid_method,
                        iou_kind,
                        iou_loss_weight,
                        objectness_loss_weight,
                        classification_loss_weight,
                    },
                momentum,
                weight_decay,
                ..
            } = config.training;

            async_std::task::spawn_blocking(move || {
                let vs = nn::VarStore::new(device);
                let root = vs.root();
                let model = yolo_dl::model::yolo_v5_small(&root, input_channels, num_classes);
                let yolo_loss = YoloLossInit {
                    reduction: Reduction::Mean,
                    match_grid_method: Some(match_grid_method),
                    iou_kind: Some(iou_kind),
                    iou_loss_weight: iou_loss_weight.map(|val| val.raw()),
                    objectness_loss_weight: objectness_loss_weight.map(|val| val.raw()),
                    classification_loss_weight: classification_loss_weight.map(|val| val.raw()),
                    ..Default::default()
                }
                .build()?;
                let optimizer = {
                    let mut opt = nn::Adam {
                        wd: weight_decay.raw(),
                        ..Default::default()
                    }
                    .build(&vs, init_lr)?;
                    opt.set_momentum(momentum.raw());
                    opt
                };

                Fallible::Ok(WorkerContext {
                    device,
                    minibatch_size,
                    vs,
                    model,
                    yolo_loss,
                    optimizer,
                })
            })
        }))
        .await?
    };
    init_timing.set_record("init worker contexts");

    // load checkpoint
    let worker_contexts_ = {
        let config = config.clone();
        async_std::task::spawn_blocking(move || -> Result<_> {
            try_load_checkpoint(
                &mut worker_contexts[0].vs,
                &config.logging.dir,
                &config.training.load_checkpoint,
            )?;
            Ok(worker_contexts)
        })
        .await?
    };
    worker_contexts = worker_contexts_;
    init_timing.set_record("load checkpoint");

    init_timing.report();

    info!("initialization finished, start training");

    // training loop
    let mut training_timing = Timing::new("training loop");

    loop {
        let record = data_rx.recv().await?;

        let TrainingRecord {
            epoch,
            step: _record_step,
            image,
            bboxes,
        } = record.to_device(master_device);
        training_timing.set_record("wait for data");

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
        training_timing.set_record("sync weights");

        // forward step
        let outputs = {
            // distribute tasks to each worker
            let jobs = {
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

                    jobs[worker_index].push(WorkerJob {
                        job_index,
                        minibatch_size,
                        image: mini_image,
                        bboxes: mini_bboxes,
                    });

                    batch_index = batch_end;
                    if batch_index == batch_size {
                        break;
                    }
                }

                jobs
            };

            // run tasks
            let (worker_contexts_, outputs_per_worker) =
                future::try_join_all(worker_contexts.into_iter().zip_eq(jobs).enumerate().map(
                    |(worker_index, (mut context, jobs))| {
                        async_std::task::spawn_blocking(move || -> Result<_> {
                            let outputs: Vec<_> = jobs
                                .into_iter()
                                .map(|job| -> Result<_> {
                                    let mut worker_timing = Timing::new("training worker");

                                    let WorkerContext {
                                        device,
                                        ref vs,
                                        ref mut model,
                                        ref yolo_loss,
                                        ref mut optimizer,
                                        ..
                                    } = context;
                                    let WorkerJob {
                                        job_index,
                                        minibatch_size,
                                        image,
                                        bboxes,
                                    } = job;

                                    let image = image.to_device(device);
                                    worker_timing.set_record("to device");

                                    // forward pass
                                    let output = model
                                        .forward_t(&image, true)?
                                        .yolo()
                                        .ok_or_else(|| format_err!("TODO"))?;
                                    worker_timing.set_record("forward");

                                    // compute loss
                                    let (losses, loss_auxiliary) =
                                        yolo_loss.forward(&output, &bboxes);
                                    worker_timing.set_record("loss");

                                    // compute gradients
                                    optimizer.zero_grad();
                                    losses.total_loss.backward();
                                    worker_timing.set_record("backward");

                                    let gradients = vs
                                        .trainable_variables()
                                        .iter()
                                        .map(|tensor| tensor.grad() * minibatch_size as f64)
                                        .collect_vec();
                                    optimizer.zero_grad();
                                    worker_timing.set_record("extract gradients");

                                    worker_timing.report();

                                    Ok(WorkerOutput {
                                        job_index,
                                        worker_index,
                                        minibatch_size,
                                        output,
                                        losses,
                                        target_bboxes: loss_auxiliary.target_bboxes,
                                        gradients,
                                    })
                                })
                                .try_collect()?;

                            Ok((context, outputs))
                        })
                    },
                ))
                .await?
                .into_iter()
                .unzip_n_vec();

            worker_contexts = worker_contexts_;

            let mut outputs = outputs_per_worker.into_iter().flatten().collect_vec();
            outputs.sort_by_cached_key(|output| output.job_index);
            outputs
        };
        training_timing.set_record("forward step");

        // backward step
        let worker_outputs = {
            let (worker_contexts_, outputs) = async_std::task::spawn_blocking(move || {
                let (worker_contexts, outputs) = tch::no_grad(|| {
                    // aggregate gradients
                    let sum_gradients = {
                        let mut gradients_iter = outputs.iter().map(|output| &output.gradients);

                        let init = gradients_iter
                            .next()
                            .unwrap()
                            .iter()
                            .map(|gradients| gradients.to_device(master_device))
                            .collect_vec();

                        let sum_gradients = gradients_iter.fold(init, |lhs, rhs| {
                            lhs.into_iter()
                                .zip_eq(rhs.iter())
                                .map(|(lhs, rhs)| lhs + rhs.to_device(master_device))
                                .collect_vec()
                        });

                        sum_gradients
                    };
                    let mean_gradients = sum_gradients
                        .into_iter()
                        .map(|mut grad| grad.g_div_1(batch_size as f64))
                        .collect_vec();

                    // optimize
                    {
                        let WorkerContext { vs, optimizer, .. } = &mut worker_contexts[0];
                        vs.trainable_variables()
                            .into_iter()
                            .zip_eq(mean_gradients)
                            .for_each(|(var, grad)| {
                                let _ = var.grad().copy_(&grad);
                            });
                        optimizer.step();
                    }

                    (worker_contexts, outputs)
                });
                (worker_contexts, outputs)
            })
            .await;
            worker_contexts = worker_contexts_;

            outputs
        };
        training_timing.set_record("backward step");

        // average losses among workers
        let (losses, worker_outputs) = async_std::task::spawn_blocking(move || -> Result<_> {
            let losses = YoloLossOutput::weighted_mean(worker_outputs.iter().map(|output| {
                (
                    output.losses.to_device(master_device),
                    output.minibatch_size as f64,
                )
            }))?;

            Ok((losses, worker_outputs))
        })
        .await?;

        training_timing.set_record("compute loss");

        // check NaN and infinite number
        tch::no_grad(|| {
            ensure!(
                bool::from(losses.total_loss.isfinite()),
                "non-finite loss detected"
            );
            Ok(())
        })?;

        // send output to logger
        if enable_training_output {
            // aggregate worker outputs
            let (model_output, target_bboxes) = {
                let (model_output_vec, target_bboxes_vec) = worker_outputs
                    .into_iter()
                    .scan(0, |batch_index_base_mut, worker_output| {
                        let WorkerOutput {
                            minibatch_size,
                            output: model_output,
                            target_bboxes: orig_target_bboxes,
                            ..
                        } = worker_output;

                        // re-index target_bboxes
                        let batch_index_base = *batch_index_base_mut;
                        let new_target_bboxes = orig_target_bboxes.into_iter().map(
                            move |(orig_instance_index, bbox)| {
                                let InstanceIndex {
                                    batch_index,
                                    layer_index,
                                    anchor_index,
                                    grid_col,
                                    grid_row,
                                } = *orig_instance_index.as_ref();
                                let new_instance_index = InstanceIndex {
                                    batch_index: batch_index + batch_index_base,
                                    layer_index,
                                    anchor_index,
                                    grid_col,
                                    grid_row,
                                };
                                (Arc::new(new_instance_index), bbox)
                            },
                        );

                        *batch_index_base_mut += minibatch_size;
                        Some((model_output, new_target_bboxes))
                    })
                    .unzip_n_vec();

                let model_output = YoloOutput::cat(model_output_vec, master_device)?;
                let target_bboxes: HashMap<_, _> =
                    target_bboxes_vec.into_iter().flatten().collect();

                assert!(target_bboxes
                    .keys()
                    .all(|index| index.batch_index < model_output.batch_size() as usize));

                (model_output, Arc::new(target_bboxes))
            };

            logging_tx
                .send(LoggingMessage::new_training_output(
                    "training-output",
                    training_step,
                    &image,
                    &model_output,
                    &losses,
                    target_bboxes,
                ))
                .map_err(|_err| format_err!("cannot send message to logger"))?;
        } else {
            logging_tx
                .send(LoggingMessage::new_training_step(
                    "loss",
                    training_step,
                    &losses,
                ))
                .map_err(|_err| format_err!("cannot send message to logger"))?;
        }

        // save checkpoint
        if let Some(0) = save_checkpoint_steps.map(|steps| training_step % steps) {
            let losses = losses.shallow_clone();
            let checkpoint_dir = checkpoint_dir.clone();

            let worker_contexts_ = async_std::task::spawn_blocking(move || -> Result<_> {
                save_checkpoint(
                    &worker_contexts[0].vs,
                    &checkpoint_dir,
                    training_step,
                    f64::from(&losses.total_loss),
                )?;

                Ok(worker_contexts)
            })
            .await?;
            worker_contexts = worker_contexts_;
            training_timing.set_record("save checkpoint");
        }

        // print message
        rate_counter.add(1.0);
        if let Some(batch_rate) = rate_counter.rate() {
            let record_rate = batch_rate * config.training.batch_size.get() as f64;
            info!(
                "epoch: {}\tstep: {}\tlr: {:.5}\t{:.2} batches/s\t{:.2} records/s",
                epoch,
                training_step,
                lr_scheduler.lr(),
                batch_rate,
                record_rate
            );
        } else {
            info!(
                "epoch: {}\tstep: {}\tlr: {:.5}",
                epoch,
                training_step,
                lr_scheduler.lr()
            );
        }

        // update lr
        {
            let lr = lr_scheduler.next();
            worker_contexts
                .iter_mut()
                .for_each(|context| context.optimizer.set_lr(lr));
        }

        training_step += 1;

        training_timing.set_record("finalize");

        {
            training_timing.report();
            training_timing = Timing::new("training loop");
        }
    }
}

fn single_gpu_training_worker(
    config: Arc<Config>,
    _logging_dir: Arc<PathBuf>,
    checkpoint_dir: Arc<PathBuf>,
    input_channels: usize,
    num_classes: usize,
    data_rx: async_std::channel::Receiver<TrainingRecord>,
    logging_tx: broadcast::Sender<LoggingMessage>,
    device: Device,
) -> Result<()> {
    info!("use single device {:?}", device);

    let Config {
        training:
            TrainingConfig {
                initial_step: init_training_step,
                ref lr_schedule,
                loss:
                    LossConfig {
                        iou_kind,
                        match_grid_method,
                        iou_loss_weight,
                        objectness_loss_weight,
                        classification_loss_weight,
                    },
                ..
            },
        logging: LoggingConfig {
            enable_training_output,
            ..
        },
        ..
    } = *config;

    // init model
    info!("initializing model");
    let mut lr_scheduler = LrScheduler::new(lr_schedule, init_training_step)?;
    let mut training_step = init_training_step;
    let mut vs = nn::VarStore::new(device);
    let root = vs.root();
    let mut model = yolo_dl::model::yolo_v5_small(&root, input_channels, num_classes);
    let yolo_loss = YoloLossInit {
        reduction: Reduction::Mean,
        match_grid_method: Some(match_grid_method),
        iou_kind: Some(iou_kind),
        iou_loss_weight: iou_loss_weight.map(|val| val.raw()),
        objectness_loss_weight: objectness_loss_weight.map(|val| val.raw()),
        classification_loss_weight: classification_loss_weight.map(|val| val.raw()),
        ..Default::default()
    }
    .build()?;
    let mut optimizer = {
        let TrainingConfig {
            momentum,
            weight_decay,
            ..
        } = config.as_ref().training;
        let lr = lr_scheduler.next();
        let mut opt = nn::Adam {
            wd: weight_decay.raw(),
            ..Default::default()
        }
        .build(&vs, lr)?;
        opt.set_momentum(momentum.raw());
        opt
    };
    let mut rate_counter = RateCounter::with_second_intertal();
    let mut timing = Timing::new("training loop");
    let save_checkpoint_steps = config
        .training
        .save_checkpoint_steps
        .map(|steps| steps.get());

    // load checkpoint
    try_load_checkpoint(
        &mut vs,
        &config.logging.dir,
        &config.training.load_checkpoint,
    )?;

    // training
    info!("start training");

    while let Ok(record) = async_std::task::block_on(data_rx.recv()) {
        timing.set_record("next record");

        let TrainingRecord {
            epoch,
            step: _record_step,
            image,
            bboxes,
        } = record.to_device(device);
        timing.set_record("to device");

        // forward pass
        let output = model
            .forward_t(&image, true)?
            .yolo()
            .ok_or_else(|| format_err!("TODO"))?;
        timing.set_record("forward");

        // compute loss
        let (losses, loss_auxiliary) = yolo_loss.forward(&output, &bboxes);
        timing.set_record("loss");

        // optimizer
        optimizer.backward_step(&losses.total_loss);
        timing.set_record("backward");

        // print message
        rate_counter.add(1.0);
        if let Some(batch_rate) = rate_counter.rate() {
            let record_rate = batch_rate * config.training.batch_size.get() as f64;
            info!(
                "epoch: {}\tstep: {}\tlr: {:.5}\t{:.2} batches/s\t{:.2} records/s",
                epoch,
                training_step,
                lr_scheduler.lr(),
                batch_rate,
                record_rate
            );
        } else {
            info!(
                "epoch: {}\tstep: {}\tlr: {:.5}",
                epoch,
                training_step,
                lr_scheduler.lr()
            );
        }

        // update lr
        optimizer.set_lr(lr_scheduler.next());

        // save checkpoint
        if let Some(0) = save_checkpoint_steps.map(|steps| training_step % steps) {
            save_checkpoint(
                &vs,
                &checkpoint_dir,
                training_step,
                f64::from(&losses.total_loss),
            )?;
        }

        // send to logger
        if enable_training_output {
            logging_tx
                .send(LoggingMessage::new_training_output(
                    "training-output",
                    training_step,
                    &image,
                    &output,
                    &losses,
                    Arc::new(loss_auxiliary.target_bboxes),
                ))
                .map_err(|_err| format_err!("cannot send message to logger"))?;
        } else {
            logging_tx
                .send(LoggingMessage::new_training_step(
                    "loss",
                    training_step,
                    &losses,
                ))
                .map_err(|_err| format_err!("cannot send message to logger"))?;
        }

        // report profiling
        {
            timing.report();
            timing = Timing::new("training loop");
        }

        training_step += 1;
    }

    Fallible::Ok(())
}

fn save_checkpoint(
    vs: &nn::VarStore,
    checkpoint_dir: &Path,
    training_step: usize,
    loss: f64,
) -> Result<()> {
    let filename = format!(
        "{}_{:06}_{:08.5}.ckpt",
        Local::now().format(FILE_STRFTIME),
        training_step,
        loss
    );
    let path = checkpoint_dir.join(filename);
    vs.save(&path)?;
    Ok(())
}

fn try_load_checkpoint(
    vs: &mut nn::VarStore,
    logging_dir: &Path,
    load_checkpoint: &LoadCheckpoint,
) -> Result<()> {
    let checkpoint_filename_regex =
        Regex::new(r"^(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.\d{3}\+\d{4})_\d{6}_\d+\.\d+\.ckpt$")
            .unwrap();

    let path = match load_checkpoint {
        LoadCheckpoint::Disabled => {
            info!("checkpoint loading is disabled");
            None
        }
        LoadCheckpoint::FromRecent => {
            let paths: Vec<_> =
                glob::glob(&format!("{}/*/checkpoints/*.ckpt", logging_dir.display()))
                    .unwrap()
                    .try_collect()?;
            let paths = paths
                .into_iter()
                .filter_map(|path| {
                    let file_name = path.file_name()?.to_str()?;
                    let captures = checkpoint_filename_regex.captures(file_name)?;
                    let datetime_str = captures.get(1)?.as_str();
                    let datetime = DateTime::parse_from_str(datetime_str, FILE_STRFTIME).unwrap();
                    Some((path, datetime))
                })
                .collect_vec();
            let checkpoint_file = paths
                .into_iter()
                .max_by_key(|(_path, datetime)| datetime.clone())
                .map(|(path, _datetime)| path);

            if let None = &checkpoint_file {
                warn!("no checkpoint file found");
            }

            checkpoint_file
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
