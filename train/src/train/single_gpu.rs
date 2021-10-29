use crate::{
    common::*,
    config,
    logging::{LoggingMessage, TrainingOutputLog},
    model::Model,
    training_stream::TrainingRecord,
    utils::{self, LrScheduler, RateCounter},
};
use tch_goodies::MergedDenseDetection;

/// Start the single-GPU training worker.
pub fn single_gpu_training_worker(
    config: ArcRef<config::Config>,
    checkpoint_dir: Arc<PathBuf>,
    mut data_rx: tokio::sync::mpsc::Receiver<TrainingRecord>,
    logging_tx: broadcast::Sender<LoggingMessage>,
    device: Device,
) -> Result<()> {
    info!("use single device {:?}", device);

    let config::Config {
        model: ref model_config,
        training:
            config::Training {
                override_initial_step,
                optimizer:
                    config::Optimizer {
                        ref lr_schedule, ..
                    },
                ..
            },
        benchmark:
            config::Benchmark {
                nms_iou_thresh,
                nms_conf_thresh,
                ..
            },
        ..
    } = *config;

    // init model
    info!("initializing model");

    const DUMMY_LR: f64 = 1.0;

    let mut vs = nn::VarStore::new(device);
    let root = vs.root();

    let mut model = Model::new(&root, model_config)?;
    let yolo_loss = config
        .training
        .loss
        .yolo_loss_init()
        .build(&root / "loss")?;
    let yolo_inference = YoloInferenceInit {
        nms_iou_thresh,
        nms_conf_thresh,
        suppress_by_class: false,
    }
    .build()?;
    let yolo_benchmark = {
        let config::Benchmark {
            nms_iou_thresh,
            nms_conf_thresh,
            ..
        } = config.benchmark;
        YoloBenchmarkInit {
            iou_threshold: nms_iou_thresh,
            confidence_threshold: nms_conf_thresh,
        }
        .build()?
    };

    let mut training_step_tensor = root.zeros_no_train("training_step", &[]);
    let mut optimizer = {
        let config::Training {
            optimizer:
                config::Optimizer {
                    momentum,
                    weight_decay,
                    ..
                },
            ..
        } = config.as_ref().training;
        let mut opt = nn::Adam {
            beta1: momentum.raw(),
            beta2: 0.999,
            wd: weight_decay.raw(),
        }
        .build(&vs, DUMMY_LR)?;
        opt.set_momentum(momentum.raw());
        opt
    };

    let save_checkpoint_steps = config
        .training
        .save_checkpoint_steps
        .map(|steps| steps.get());

    // load checkpoint
    let init_training_step = {
        utils::try_load_checkpoint(
            &mut vs,
            &config.logging.dir,
            &config.training.load_checkpoint,
        )?;

        match override_initial_step {
            Some(init_step) => {
                training_step_tensor.copy_(&Tensor::from(init_step as f32));
                init_step
            }
            None => match &config.training.load_checkpoint {
                config::LoadCheckpoint::Disabled => 0,
                _ => f32::from(&training_step_tensor) as usize + 1,
            },
        }
    };

    // training
    {
        info!("start training");
        let mut training_step = init_training_step;
        let mut rate_counter = RateCounter::with_second_intertal();
        let mut lr_scheduler = LrScheduler::new(lr_schedule, init_training_step)?;
        let clip_grad = config.training.optimizer.clip_grad.map(|val| val.raw());

        // set init lr
        {
            let init_lr = lr_scheduler.next();
            optimizer.set_lr(init_lr);
        }

        loop {
            let mut record = data_rx.blocking_recv().unwrap();
            record.timing.add_event("in channel");

            let TrainingRecord {
                epoch,
                step: _record_step,
                image,
                bboxes,
                mut timing,
            } = record.to_device(device);
            timing.add_event("move to master device");

            // forward pass
            let output = model.forward_t(&image, true)?;
            let output = MergedDenseDetection::try_from(output)?;
            timing.add_event("forward");

            // compute loss
            let (losses, loss_auxiliary) = yolo_loss.forward(&output, &bboxes);
            timing.add_event("loss");

            // optimizer
            match clip_grad {
                Some(clip_grad) => {
                    optimizer.backward_step_clip(&losses.total_loss, clip_grad);
                }
                None => {
                    optimizer.backward_step(&losses.total_loss);
                }
            }
            timing.add_event("backward");

            // clip batch norm
            model.clamp_running_var();

            // run inference
            let inference = {
                let config::Logging {
                    enable_inference,
                    enable_benchmark,
                    ..
                } = config.logging;
                (enable_inference || enable_benchmark).then(|| {
                    let inference = yolo_inference.forward(&output);
                    timing.add_event("inference");
                    inference
                })
            };

            // run benchmark
            let benchmark = config.logging.enable_benchmark.then(|| {
                let benchmark = yolo_benchmark.forward(
                    &output,
                    &loss_auxiliary.matchings,
                    inference.as_ref().unwrap(),
                );
                benchmark
            });

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

            // save checkpoint
            if let Some(0) = save_checkpoint_steps.map(|steps| training_step % steps) {
                utils::save_checkpoint(
                    &vs,
                    &checkpoint_dir,
                    training_step,
                    f64::from(&losses.total_loss),
                )?;
            }

            // save weights
            let (weights, gradients) = if config.logging.enable_gradients {
                let mut vars: Vec<_> = vs.variables().into_iter().collect();
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

            // send to logger

            logging_tx
                .send(
                    TrainingOutputLog {
                        step: training_step,
                        lr: r64(lr_scheduler.lr()),
                        input: image,
                        output,
                        losses,
                        matchings: loss_auxiliary.matchings,
                        inference,
                        benchmark,
                        weights,
                        gradients,
                    }
                    .into(),
                )
                .map_err(|_err| format_err!("cannot send message to logger"))?;

            // update lr
            optimizer.set_lr(lr_scheduler.next());

            // update training step
            training_step += 1;
            training_step_tensor.copy_(&Tensor::from(training_step as f32));

            // report profiling
            timing.report();
        }
    }
}
