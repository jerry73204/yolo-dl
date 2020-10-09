mod common;
mod config;
mod data;
mod logging;
mod message;
mod util;

use crate::{
    common::*,
    config::{Config, LoadCheckpoint, TrainingConfig},
    data::{DataSet, TrainingRecord},
    message::LoggingMessage,
    util::{LrScheduler, RateCounter, Timing},
};

const FILE_STRFTIME: &str = "%Y-%m-%d-%H-%M-%S.%3f%z";

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

    // load dataset
    info!("loading dataset");
    let dataset = DataSet::new(config.clone()).await?;
    let input_channels = dataset.input_channels();
    let num_classes = dataset.num_classes();

    // create channels
    let (logging_tx, logging_rx) = broadcast::channel(2);
    let (training_tx, training_rx) = async_std::sync::channel(2);

    // start logger
    let logging_future =
        logging::logging_worker(config.clone(), logging_dir.clone(), logging_rx).await?;

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
        let logging_dir = logging_dir.clone();
        let checkpoint_dir = checkpoint_dir.clone();

        if config.training.enable_multi_gpu {
            async_std::task::spawn(multi_gpu_training_worker(
                config,
                logging_dir,
                checkpoint_dir,
                input_channels,
                num_classes,
                training_rx,
                logging_tx,
            ))
        } else {
            async_std::task::spawn_blocking(move || {
                single_gpu_training_worker(
                    config,
                    logging_dir,
                    checkpoint_dir,
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
    logging_dir: Arc<PathBuf>,
    checkpoint_dir: Arc<PathBuf>,
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
        gradients: Vec<Tensor>,
    }

    let mut lr_scheduler = LrScheduler::new(&config.training.lr_schedule)?;
    let mut rate_counter = RateCounter::with_second_intertal();
    let master_device = config.training.master_device;
    let batch_size = config.training.batch_size.get();
    let default_minibatch_size = config.training.default_minibatch_size.get();
    let save_checkpoint_steps = config
        .training
        .save_checkpoint_steps
        .map(|steps| steps.get());

    // initialize workers
    let mut worker_contexts: Vec<_> = {
        let init_lr = lr_scheduler.next();
        config
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
                    reduction: Reduction::Mean,
                    match_grid_method: Some(config.training.match_grid_method),
                    iou_kind: Some(config.training.iou_kind),
                    ..Default::default()
                }
                .build();
                let optimizer = {
                    let TrainingConfig {
                        momentum,
                        weight_decay,
                        ..
                    } = config.as_ref().training;
                    let mut opt = nn::Adam {
                        wd: weight_decay.raw(),
                        ..Default::default()
                    }
                    .build(&vs, init_lr)?;
                    opt.set_momentum(momentum.raw());
                    opt
                };

                Ok(WorkerContext {
                    device,
                    minibatch_size,
                    vs,
                    model,
                    yolo_loss,
                    optimizer,
                })
            })
            .try_collect()?
    };

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

    // training loop
    let mut training_step = config.training.initial_step.unwrap_or(0);

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

            // run tasks
            let (worker_contexts_, outputs_per_worker) =
                future::join_all(worker_contexts.into_iter().zip_eq(jobs).enumerate().map(
                    |(worker_index, (mut context, jobs))| {
                        async_std::task::spawn_blocking(move || {
                            let outputs = jobs
                                .into_iter()
                                .map(|job| {
                                    let mut timing = Timing::new();

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
                                    timing.set_record("to device");

                                    // forward pass
                                    let output = model.forward_t(&image, true);
                                    timing.set_record("forward");

                                    // compute loss
                                    let losses = yolo_loss.forward(&output, &bboxes);
                                    timing.set_record("loss");

                                    // compute gradients
                                    optimizer.zero_grad();
                                    losses.loss.backward();
                                    timing.set_record("run_backward");

                                    let gradients = vs
                                        .trainable_variables()
                                        .iter()
                                        .map(|tensor| tensor.grad() * minibatch_size as f64)
                                        .collect_vec();

                                    WorkerOutput {
                                        job_index,
                                        worker_index,
                                        minibatch_size,
                                        output,
                                        losses,
                                        gradients,
                                    }
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
            outputs.sort_by_cached_key(|output| output.job_index);
            outputs
        };

        // backward step
        let outputs = {
            let (worker_contexts_, outputs) = async_std::task::spawn_blocking(move || {
                tch::no_grad(|| {
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
                })
            })
            .await;
            worker_contexts = worker_contexts_;

            outputs
        };

        // compute losses from each job
        let losses = async_std::task::spawn_blocking(move || {
            let (
                loss_vec,
                iou_loss_vec,
                classification_loss_vec,
                objectness_loss_vec,
                target_bboxes_vec,
            ) = outputs
                .iter()
                .map(|output| {
                    let YoloLossOutput {
                        loss,
                        iou_loss,
                        classification_loss,
                        objectness_loss,
                        target_bboxes,
                    } = &output.losses;
                    let minibatch_size = output.minibatch_size as f64;
                    (
                        loss * minibatch_size,
                        iou_loss * minibatch_size,
                        classification_loss * minibatch_size,
                        objectness_loss * minibatch_size,
                        target_bboxes,
                    )
                })
                .unzip_n_vec();

            let mean_tensors = |tensors: &[Tensor]| -> Tensor {
                let mut iter = tensors.iter();
                let first = iter.next().unwrap().to_device(master_device);
                iter.fold(first, |lhs, rhs| lhs + rhs.to_device(master_device)) / batch_size as f64
            };

            let loss = mean_tensors(&loss_vec);
            let iou_loss = mean_tensors(&iou_loss_vec);
            let classification_loss = mean_tensors(&classification_loss_vec);
            let objectness_loss = mean_tensors(&objectness_loss_vec);

            // WARNING: it is incorrect but compiles
            let target_bboxes: HashMap<_, _> = target_bboxes_vec
                .into_iter()
                .flat_map(|bboxes| {
                    bboxes
                        .iter()
                        .map(|(instance_index, bbox)| (instance_index.clone(), bbox.clone()))
                })
                .collect();

            let losses = YoloLossOutput {
                loss,
                iou_loss,
                classification_loss,
                objectness_loss,
                target_bboxes: Arc::new(target_bboxes),
            };

            losses
        })
        .await;

        // save checkpoint
        if let Some(0) = save_checkpoint_steps.map(|steps| training_step % steps) {
            let losses = losses.shallow_clone();
            let checkpoint_dir = checkpoint_dir.clone();

            let worker_contexts_ = async_std::task::spawn_blocking(move || -> Result<_> {
                save_checkpoint(
                    &worker_contexts[0].vs,
                    &checkpoint_dir,
                    training_step,
                    f64::from(&losses.loss),
                )?;

                Ok(worker_contexts)
            })
            .await?;
            worker_contexts = worker_contexts_;
        }

        // send to logger
        {
            let msg = LoggingMessage::new_training_step("loss", training_step, &losses);
            logging_tx
                .send(msg)
                .map_err(|_err| format_err!("cannot send message to logger"))?;
        }

        // {
        //     let msg = LoggingMessage::new_training_output(
        //         "output",
        //         training_step,
        //         &image,
        //         &output,
        //         &losses,
        //     );
        //     logging_tx
        //         .send(msg)
        //         .map_err(|_err| format_err!("cannot send message to logger"))?;
        // }

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
                "epoch: {}\tstep: {}\tlr: {:5}",
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
    }
    Ok(())
}

fn single_gpu_training_worker(
    config: Arc<Config>,
    logging_dir: Arc<PathBuf>,
    checkpoint_dir: Arc<PathBuf>,
    input_channels: usize,
    num_classes: usize,
    training_rx: async_std::sync::Receiver<TrainingRecord>,
    logging_tx: broadcast::Sender<LoggingMessage>,
) -> Result<()> {
    warn!(r#"multi-gpu feature is disabled, use "master_device" as default device"#);

    // init model
    info!("initializing model");
    let device = config.training.master_device;
    let mut lr_scheduler = LrScheduler::new(&config.training.lr_schedule)?;
    let mut vs = nn::VarStore::new(device);
    let root = vs.root();
    let mut model = yolo_dl::model::yolo_v5_small(&root, input_channels, num_classes);
    let yolo_loss = YoloLossInit {
        match_grid_method: Some(config.training.match_grid_method),
        iou_kind: Some(config.training.iou_kind),
        ..Default::default()
    }
    .build();
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
    let mut timing = Timing::new();
    let mut training_step = config.training.initial_step.unwrap_or(0);
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
                "epoch: {}\tstep: {}\tlr: {:.5}\t{:.2} batches/s\t{:.2} records/s",
                epoch,
                training_step,
                lr_scheduler.lr(),
                batch_rate,
                record_rate
            );
        } else {
            info!(
                "epoch: {}\tstep: {}\tlr: {:5}",
                epoch,
                training_step,
                lr_scheduler.lr()
            );
        }

        // update lr
        optimizer.set_lr(lr_scheduler.next());

        // save checkpoint
        if let Some(0) = save_checkpoint_steps.map(|steps| training_step % steps) {
            save_checkpoint(&vs, &checkpoint_dir, training_step, f64::from(&losses.loss))?;
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
