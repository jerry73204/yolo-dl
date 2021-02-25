use crate::{
    common::*,
    config::{Config, LoadCheckpoint, LossConfig, TrainingConfig},
    data::TrainingRecord,
    logging::LoggingMessage,
    model::Model,
    utils::{self, LrScheduler, RateCounter},
};

struct WorkerContext {
    device: Device,
    minibatch_size: usize,
    vs: nn::VarStore,
    model: Model,
    yolo_loss: YoloLoss,
    optimizer: nn::Optimizer<nn::Adam>,
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
    output: MergeDetect2DOutput,
    losses: YoloLossOutput,
    loss_auxiliary: YoloLossAuxiliary,
    gradients: Vec<Tensor>,
}

/// Start the multi-GPU training worker.
pub async fn multi_gpu_training_worker(
    config: Arc<Config>,
    _logging_dir: Arc<PathBuf>,
    checkpoint_dir: Arc<PathBuf>,
    _input_channels: usize,
    _num_classes: usize,
    data_rx: async_std::channel::Receiver<TrainingRecord>,
    logging_tx: broadcast::Sender<LoggingMessage>,
    workers: &[(Device, usize)],
) -> Result<()> {
    ensure!(!workers.is_empty(), "worker list must not be empty");

    // initialization
    info!(
        "use device configuration (device, minibatch_size): {:?}",
        workers
    );

    // initialize workers
    let (mut worker_contexts, init_training_step) =
        initialize_worker_contexts(config.clone(), workers).await?;

    info!("initialization finished, start training");

    // training loop
    {
        let Config {
            training:
                TrainingConfig {
                    save_checkpoint_steps,
                    ref lr_schedule,
                    ..
                },
            ..
        } = *config;
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
            let mut record = data_rx.recv().await?;
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
                    backward_step(&config, master_device, &mut worker_contexts, &outputs)?;
                    Ok((worker_contexts, outputs))
                })
                .map(|result| Fallible::Ok(result??))
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

                Ok((losses, worker_outputs))
            })
            .map(|result| Fallible::Ok(result??))
            .await?;

            // check NaN and infinite number
            tch::no_grad(|| {
                ensure!(
                    bool::from(losses.total_loss.isfinite()),
                    "non-finite loss detected"
                );
                Ok(())
            })?;

            timing.add_event("compute loss");

            // merge output
            let (model_output, target_bboxes) = {
                let (model_output_vec, target_bboxes_vec) = worker_outputs
                    .iter()
                    .scan(0, |batch_index_base_mut, worker_output| {
                        let WorkerOutput {
                            minibatch_size,
                            output: ref model_output,
                            ref loss_auxiliary,
                            ..
                        } = *worker_output;

                        // re-index target_bboxes
                        let batch_index_base = *batch_index_base_mut;
                        let new_target_bboxes = loss_auxiliary.target_bboxes.0.iter().map(
                            move |(instance_index, bbox)| {
                                let new_instance_index = InstanceIndex {
                                    batch_index: instance_index.batch_index + batch_index_base,
                                    ..*instance_index
                                };
                                (new_instance_index, bbox.to_owned())
                            },
                        );

                        *batch_index_base_mut += minibatch_size;
                        Some((model_output, new_target_bboxes))
                    })
                    .unzip_n_vec();

                let model_output = MergeDetect2DOutput::cat(model_output_vec, master_device)?;
                let target_bboxes =
                    PredTargetMatching(target_bboxes_vec.into_iter().flatten().collect());

                (model_output, target_bboxes)
            };

            timing.add_event("merge outputs");

            // compute benchmark
            // {
            //     let benchmark = YoloBenchmarkInit::default().build()?;
            //     benchmark.forward(&model_output);
            // }

            // send output to logger
            {
                let losses = losses.shallow_clone();
                log_outputs(
                    config.clone(),
                    logging_tx.clone(),
                    training_step,
                    image,
                    model_output,
                    target_bboxes,
                    losses,
                )
                .await?;
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
                .map(|result| Fallible::Ok(result??))
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
}

async fn initialize_worker_contexts(
    config: Arc<Config>,
    workers: &[(Device, usize)],
) -> Result<(Vec<WorkerContext>, usize)> {
    const DUMMY_LR: f64 = 1.0;

    let mut init_timing = Timing::new("initialization");

    // initialize workers
    info!("initializing model");

    let mut worker_contexts: Vec<_> = {
        let config = config.clone();

        stream::iter(workers.to_owned())
            .wrapping_enumerate()
            .par_map_unordered(None, move |(index, (device, minibatch_size))| {
                let config = config.clone();

                move || {
                    let TrainingConfig {
                        loss:
                            LossConfig {
                                match_grid_method,
                                box_metric,
                                iou_loss_weight,
                                objectness_loss_weight,
                                classification_loss_weight,
                            },
                        momentum,
                        weight_decay,
                        ..
                    } = config.training;
                    let model_config = config.model.clone();

                    let vs = nn::VarStore::new(device);
                    let root = vs.root();

                    let model = Model::new(&root, &model_config)?;
                    let yolo_loss = YoloLossInit {
                        reduction: Reduction::Mean,
                        match_grid_method: Some(match_grid_method),
                        box_metric: Some(box_metric),
                        iou_loss_weight: iou_loss_weight.map(R64::raw),
                        objectness_loss_weight: objectness_loss_weight.map(R64::raw),
                        classification_loss_weight: classification_loss_weight.map(R64::raw),
                        ..Default::default()
                    }
                    .build()?;

                    let training_step = root.zeros_no_train("training_step", &[]);

                    let optimizer = {
                        let mut opt = nn::Adam {
                            wd: weight_decay.raw(),
                            ..Default::default()
                        }
                        .build(&vs, DUMMY_LR)?;
                        opt.set_momentum(momentum.raw());
                        opt
                    };

                    Fallible::Ok((
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
        .map(|result| Fallible::Ok(result??))
        .await?
    };
    init_timing.add_event("load checkpoint");

    // load initial training step
    let init_training_step = {
        let Config {
            training:
                TrainingConfig {
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
                LoadCheckpoint::Disabled => 0,
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
            .wrapping_enumerate()
            .par_map(None, move |(index, mut context)| {
                let first_vs = first_vs.clone();
                move || {
                    context.vs.copy(&*first_vs)?;
                    Fallible::Ok((index, context))
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
    config: Arc<Config>,
    worker_contexts: Vec<WorkerContext>,
    image: Tensor,
    bboxes: &[Vec<RatioLabel>],
) -> Result<(Vec<WorkerContext>, Vec<WorkerOutput>)> {
    let Config {
        training: TrainingConfig { batch_size, .. },
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
            .wrapping_enumerate()
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

                    Fallible::Ok((worker_index, (context, outputs)))
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
    config: &Config,
    master_device: Device,
    worker_contexts: &mut [WorkerContext],
    outputs: &[WorkerOutput],
) -> Result<()> {
    tch::no_grad(|| {
        let Config {
            training: TrainingConfig { batch_size, .. },
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

        Ok(())
    })
}

async fn log_outputs(
    _config: Arc<Config>,
    logging_tx: broadcast::Sender<LoggingMessage>,
    training_step: usize,
    image: Tensor,
    model_output: MergeDetect2DOutput,
    target_bboxes: PredTargetMatching,
    losses: YoloLossOutput,
) -> Result<()> {
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

    Ok(())
}
