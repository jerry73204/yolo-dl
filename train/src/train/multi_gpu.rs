use tch_goodies::{lr_schedule::LrScheduler, MergedDenseDetection};

use crate::{
    common::*,
    config,
    logging::{LoggingMessage, TrainingOutputLog},
    model::Model,
    training_stream::TrainingRecord,
    utils::{self, RateCounter},
};
use yolo_dl::{
    label::RatioLabel,
    loss::{
        MatchingOutput, YoloBenchmarkInit, YoloInferenceInit, YoloLoss, YoloLossAuxiliary,
        YoloLossOutput,
    },
    profiling::Timing,
};

struct WorkerContext {
    device: Device,
    minibatch_size: usize,
    vs: nn::VarStore,
    model: Model,
    yolo_loss: YoloLoss,
    optimizer: nn::Optimizer,
    training_step: Tensor,
}

struct WorkerJob {
    job_index: usize,
    minibatch_size: usize,
    image: Tensor,
    bboxes: Vec<Vec<RatioLabel>>,
}

struct WorkerOutput {
    job_index: usize,
    worker_index: usize,
    minibatch_size: usize,
    output: MergedDenseDetection,
    losses: YoloLossOutput,
    loss_auxiliary: YoloLossAuxiliary,
    gradients: Vec<Tensor>,
}

/// Start the multi-GPU training worker.
pub async fn multi_gpu_training_worker(
    config: ArcRef<config::Config>,
    checkpoint_dir: Arc<PathBuf>,
    mut data_rx: flume::Receiver<TrainingRecord>,
    logging_tx: broadcast::Sender<LoggingMessage>,
    workers: &[(Device, usize)],
) -> Result<()> {
    ensure!(!workers.is_empty(), "worker list must not be empty");

    // initialization
    info!(
        "use device configuration (device, minibatch_size): {:?}",
        workers
    );

    let yolo_inference = {
        let config::Benchmark {
            nms_iou_thresh,
            nms_conf_thresh,
            ..
        } = config.benchmark;
        let yolo_inference = YoloInferenceInit {
            nms_iou_thresh,
            nms_conf_thresh,
            suppress_by_class: false,
        }
        .build()?;
        Arc::new(yolo_inference)
    };
    let yolo_benchmark = {
        let config::Benchmark {
            nms_iou_thresh,
            nms_conf_thresh,
            ..
        } = config.benchmark;
        let yolo_benchmark = YoloBenchmarkInit {
            iou_threshold: nms_iou_thresh,
            confidence_threshold: nms_conf_thresh,
        }
        .build()?;
        Arc::new(yolo_benchmark)
    };

    // initialize workers
    let (mut worker_contexts, init_training_step) =
        initialize_worker_contexts(config.clone(), workers).await?;

    info!("initialization finished, start training");

    // training loop
    {
        let config::Config {
            training:
                config::Training {
                    save_checkpoint_steps,
                    optimizer:
                        config::Optimizer {
                            ref lr_schedule,
                            clip_grad,
                            ..
                        },
                    ..
                },
            ..
        } = *config;
        let clip_grad = clip_grad.map(|val| val.raw());
        let save_checkpoint_steps = save_checkpoint_steps.map(|steps| steps.get());
        if let None = save_checkpoint_steps {
            warn!("checkpoint saving is disabled");
        }
        let (master_device, _) = workers[0];

        let mut rate_counter = RateCounter::with_second_intertal();
        let mut training_step = init_training_step;
        let mut lr_scheduler = LrScheduler::new(lr_schedule, init_training_step)?;

        // update initial learning rate
        {
            let init_lr = lr_scheduler.next();
            worker_contexts.iter_mut().for_each(|context| {
                context.optimizer.set_lr(init_lr);
            });
        }

        loop {
            let mut record = match data_rx.recv_async().await {
                Ok(record) => record,
                Err(_) => break,
            };
            record.timing.add_event("in channel");

            let (epoch, image, bboxes, mut timing) = tokio::task::spawn_blocking(move || {
                let TrainingRecord {
                    epoch,
                    image,
                    bboxes,
                    timing,
                    ..
                } = record;

                // changing device is expensive
                let image = image.to_device(master_device);

                (epoch, image, bboxes, timing)
            })
            .await?;
            timing.add_event("move to master device");

            // sync weights among workers
            worker_contexts = sync_weights(worker_contexts).await?;
            timing.add_event("sync weights");

            // forward step
            let (worker_contexts_, outputs) = {
                let image = image.shallow_clone();
                forward_step(config.clone(), worker_contexts, image, &bboxes).await?
            };
            worker_contexts = worker_contexts_;
            timing.add_event("forward step");

            // backward step
            let (worker_contexts_, worker_outputs) = {
                let config = config.clone();
                tokio::task::spawn_blocking(move || -> Result<_> {
                    backward_step(
                        &config,
                        master_device,
                        &mut worker_contexts,
                        &outputs,
                        clip_grad,
                    )?;
                    Ok((worker_contexts, outputs))
                })
                .map(|result| anyhow::Ok(result??))
                .await?
            };
            worker_contexts = worker_contexts_;
            timing.add_event("backward step");

            // merge outputs and losses
            let (losses, worker_outputs) = tokio::task::spawn_blocking(move || -> Result<_> {
                let weighted_outputs = worker_outputs.iter().map(|output| {
                    (
                        output.losses.to_device(master_device),
                        output.minibatch_size as f64,
                    )
                });
                let losses = YoloLossOutput::weighted_mean(weighted_outputs)?;

                // check NaN and infinite number
                tch::no_grad(|| {
                    ensure!(
                        bool::from(losses.total_loss.isfinite()),
                        "non-finite loss detected"
                    );
                    Ok(())
                })?;

                Ok((losses, worker_outputs))
            })
            .map(|result| anyhow::Ok(result??))
            .await?;

            timing.add_event("compute loss");

            let (model_output, matchings, inference, benchmark, mut timing) = {
                let config = config.clone();
                let yolo_benchmark = yolo_benchmark.clone();
                let yolo_inference = yolo_inference.clone();

                tokio::task::spawn_blocking(move || -> Result<_> {
                    // merge output
                    let model_output = MergedDenseDetection::cat(
                        worker_outputs
                            .iter()
                            .map(|output| output.output.to_device(master_device)),
                    )?;
                    let matchings = MatchingOutput::cat_mini_batches(worker_outputs.iter().map(
                        |worker_output| {
                            let WorkerOutput {
                                minibatch_size,
                                loss_auxiliary: YoloLossAuxiliary { ref matchings, .. },
                                ..
                            } = *worker_output;
                            (matchings.to_device(master_device), minibatch_size as i64)
                        },
                    ));

                    timing.add_event("merge outputs");

                    // run inference
                    let inference = {
                        let config::Logging {
                            enable_inference,
                            enable_benchmark,
                            ..
                        } = config.logging;
                        (enable_inference || enable_benchmark).then(|| {
                            let inference = yolo_inference.forward(&model_output);
                            timing.add_event("inference");
                            inference
                        })
                    };

                    // run benchmark
                    let benchmark = config.logging.enable_inference.then(|| {
                        let benchmark = yolo_benchmark.forward(
                            &model_output,
                            &matchings,
                            inference.as_ref().unwrap(),
                        );
                        timing.add_event("benchmark");
                        benchmark
                    });

                    Ok((model_output, matchings, inference, benchmark, timing))
                })
                .await??
            };

            // save weights and gradients
            let (weights, gradients) = if config.logging.enable_gradients {
                let mut vars: Vec<_> = worker_contexts[0].vs.variables().into_iter().collect();
                vars.sort_by_cached_key(|(name, _var)| name.to_owned());
                let weights: Vec<_> = vars
                    .iter()
                    .map(|(name, var)| {
                        let max = f64::from(var.abs().max());
                        (name.to_owned(), max)
                    })
                    .collect();
                let grads: Vec<_> = vars
                    .iter()
                    .filter(|(_name, var)| var.requires_grad())
                    .map(|(name, var)| {
                        let max = f64::from(var.grad().abs().max());
                        (name.to_owned(), max)
                    })
                    .collect();
                (Some(weights), Some(grads))
            } else {
                (None, None)
            };

            // send output to logger
            {
                let losses = losses.shallow_clone();
                logging_tx
                    .send(
                        TrainingOutputLog {
                            step: training_step,
                            lr: r64(lr_scheduler.lr()),
                            input: image,
                            output: model_output.into(),
                            losses,
                            matchings,
                            inference,
                            benchmark,
                            weights,
                            gradients,
                        }
                        .into(),
                    )
                    .map_err(|_err| format_err!("cannot send message to logger"))?;
            }

            // save checkpoint
            let total_loss = f64::from(&losses.total_loss);

            if let Some(0) = save_checkpoint_steps.map(|steps| training_step % steps) {
                let checkpoint_dir = checkpoint_dir.clone();

                worker_contexts = tokio::task::spawn_blocking(move || -> Result<_> {
                    utils::save_checkpoint(
                        &worker_contexts[0].vs,
                        &checkpoint_dir,
                        training_step,
                        total_loss,
                    )?;

                    Ok(worker_contexts)
                })
                .map(|result| anyhow::Ok(result??))
                .await?;
                timing.add_event("save checkpoint");
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

            // update learning rate
            {
                let lr = lr_scheduler.next();
                worker_contexts
                    .iter_mut()
                    .for_each(|context| context.optimizer.set_lr(lr));
            }

            // update training step
            training_step += 1;
            worker_contexts.iter_mut().for_each(|context| {
                context
                    .training_step
                    .copy_(&Tensor::from(training_step as f32))
            });

            timing.add_event("finalize");

            // report elapsed time
            timing.report();
        }
    }

    Ok(())
}

async fn initialize_worker_contexts(
    config: ArcRef<config::Config>,
    workers: &[(Device, usize)],
) -> Result<(Vec<WorkerContext>, usize)> {
    const DUMMY_LR: f64 = 1.0;

    let mut init_timing = Timing::new("initialization");

    // initialize workers
    info!("initializing model");

    let mut worker_contexts: Vec<_> = {
        let config = config.clone();

        stream::iter(workers.to_owned())
            .enumerate()
            .par_map_unordered(None, move |(index, (device, minibatch_size))| {
                let config = config.clone();

                move || {
                    let config::Training {
                        optimizer:
                            config::Optimizer {
                                momentum,
                                weight_decay,
                                ..
                            },
                        ..
                    } = config.training;
                    let model_config = config.model.clone();

                    let vs = nn::VarStore::new(device);
                    let root = vs.root();

                    let model = Model::new(&root, &model_config)?;
                    let yolo_loss = config
                        .training
                        .loss
                        .yolo_loss_init()
                        .build(&root / "loss")?;

                    let training_step = root.zeros_no_train("training_step", &[]);

                    let optimizer = {
                        let mut opt = nn::Adam {
                            beta1: momentum.raw(),
                            beta2: 0.999,
                            wd: weight_decay.raw(),
                        }
                        .build(&vs, DUMMY_LR)?;
                        opt.set_momentum(momentum.raw());
                        opt
                    };

                    anyhow::Ok((
                        index,
                        WorkerContext {
                            device,
                            minibatch_size,
                            vs,
                            model,
                            yolo_loss,
                            optimizer,
                            training_step,
                        },
                    ))
                }
            })
            .try_reorder_enumerated()
            .try_collect()
            .await?
    };

    init_timing.add_event("init worker contexts");

    // load checkpoint (to first worker)
    worker_contexts = {
        let config = config.clone();
        tokio::task::spawn_blocking(move || -> Result<_> {
            utils::try_load_checkpoint(
                &mut worker_contexts[0].vs,
                &config.logging.dir,
                &config.training.load_checkpoint,
            )?;

            Ok(worker_contexts)
        })
        .map(|result| anyhow::Ok(result??))
        .await?
    };
    init_timing.add_event("load checkpoint");

    // load initial training step
    let init_training_step = {
        let config::Config {
            training:
                config::Training {
                    override_initial_step,
                    ..
                },
            ..
        } = *config;
        let training_step_tensor = &mut worker_contexts[0].training_step;

        match override_initial_step {
            Some(init_step) => {
                training_step_tensor.copy_(&Tensor::from(init_step as f32));
                init_step
            }
            None => match &config.training.load_checkpoint {
                config::LoadCheckpoint::Disabled => 0,
                _ => f32::from(&*training_step_tensor) as usize + 1,
            },
        }
    };

    init_timing.report();

    Ok((worker_contexts, init_training_step))
}

async fn sync_weights(worker_contexts: Vec<WorkerContext>) -> Result<Vec<WorkerContext>> {
    let mut iter = worker_contexts.into_iter();
    let first_context = iter.next().unwrap();
    let first_vs = Arc::new(first_context.vs);
    let other_contexts: Vec<_> = {
        let first_vs = first_vs.clone();
        stream::iter(iter)
            .enumerate()
            .par_map(None, move |(index, mut context)| {
                let first_vs = first_vs.clone();
                move || {
                    context.vs.copy(&*first_vs)?;
                    anyhow::Ok((index, context))
                }
            })
            .try_reorder_enumerated()
            .try_collect()
            .await?
    };

    let first_context = WorkerContext {
        vs: Arc::try_unwrap(first_vs).unwrap(),
        ..first_context
    };

    let worker_contexts: Vec<_> = iter::once(first_context).chain(other_contexts).collect();
    Ok(worker_contexts)
}

async fn forward_step(
    config: ArcRef<config::Config>,
    worker_contexts: Vec<WorkerContext>,
    image: Tensor,
    bboxes: &[Vec<RatioLabel>],
) -> Result<(Vec<WorkerContext>, Vec<WorkerOutput>)> {
    let config::Config {
        training: config::Training { batch_size, .. },
        ..
    } = *config;
    let batch_size = batch_size.get();

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
    let (worker_contexts, outputs_per_worker) =
        stream::iter(worker_contexts.into_iter().zip_eq(jobs))
            .enumerate()
            .par_map_unordered(None, |(worker_index, (mut context, jobs))| {
                move || {
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
                            worker_timing.add_event("to device");

                            // forward pass
                            let output = model.forward_t(&image, true)?;
                            let output = MergedDenseDetection::try_from(output)?;
                            worker_timing.add_event("forward");

                            // compute loss
                            let (losses, loss_auxiliary) = yolo_loss.forward(&output, &bboxes);
                            worker_timing.add_event("loss");

                            // compute gradients
                            optimizer.zero_grad();
                            losses.total_loss.backward();
                            worker_timing.add_event("backward");

                            let gradients = vs
                                .trainable_variables()
                                .iter()
                                .map(|tensor| tensor.grad() * minibatch_size as f64)
                                .collect_vec();
                            optimizer.zero_grad();
                            worker_timing.add_event("extract gradients");

                            worker_timing.report();

                            Ok(WorkerOutput {
                                job_index,
                                worker_index,
                                minibatch_size,
                                output,
                                losses,
                                loss_auxiliary,
                                gradients,
                            })
                        })
                        .try_collect()?;

                    anyhow::Ok((worker_index, (context, outputs)))
                }
            })
            .try_reorder_enumerated()
            .try_collect::<Vec<_>>()
            .await?
            .into_iter()
            .unzip_n_vec();

    let mut outputs = outputs_per_worker.into_iter().flatten().collect_vec();
    outputs.sort_by_cached_key(|output| output.job_index);

    Ok((worker_contexts, outputs))
}

fn backward_step(
    config: &config::Config,
    master_device: Device,
    worker_contexts: &mut [WorkerContext],
    outputs: &[WorkerOutput],
    clip_grad: Option<f64>,
) -> Result<()> {
    tch::no_grad(|| {
        let config::Config {
            training: config::Training { batch_size, .. },
            ..
        } = *config;
        let batch_size = batch_size.get();

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
            .map(|grad| grad.g_div_scalar(batch_size as f64))
            .collect_vec();

        // optimize
        {
            let WorkerContext {
                vs,
                optimizer,
                model,
                ..
            } = &mut worker_contexts[0];

            // copy gradients
            vs.trainable_variables()
                .into_iter()
                .zip_eq(mean_gradients)
                .for_each(|(var, grad)| {
                    let _ = var.grad().copy_(&grad);
                });

            // clip gradient
            if let Some(clip_grad) = clip_grad {
                optimizer.clip_grad_value(clip_grad);
            }

            // optimize
            optimizer.step();

            // clamp batch norm
            model.clamp_running_var();
        }

        Ok(())
    })
}
