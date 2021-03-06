use crate::{
    common::*,
    config::{Config, LoadCheckpoint, LossConfig, OptimizerConfig, TrainingConfig},
    data::TrainingRecord,
    logging::{LoggingMessage, TrainingOutputLog},
    model::Model,
    utils::{self, LrScheduler, RateCounter},
};

/// Start the single-GPU training worker.
pub fn single_gpu_training_worker(
    config: Arc<Config>,
    _logging_dir: Arc<PathBuf>,
    checkpoint_dir: Arc<PathBuf>,
    _input_channels: usize,
    _num_classes: usize,
    data_rx: async_std::channel::Receiver<TrainingRecord>,
    logging_tx: broadcast::Sender<LoggingMessage>,
    device: Device,
) -> Result<()> {
    info!("use single device {:?}", device);

    let Config {
        model: ref model_config,
        training:
            TrainingConfig {
                override_initial_step,
                optimizer:
                    OptimizerConfig {
                        ref lr_schedule, ..
                    },
                loss:
                    LossConfig {
                        box_metric,
                        match_grid_method,
                        iou_loss_weight,
                        objectness_positive_weight,
                        objectness_loss_fn,
                        classification_loss_fn,
                        objectness_loss_weight,
                        classification_loss_weight,
                    },
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
    let yolo_loss = YoloLossInit {
        reduction: Reduction::Mean,
        match_grid_method: Some(match_grid_method),
        box_metric: Some(box_metric),
        iou_loss_weight: iou_loss_weight.map(|val| val.raw()),
        objectness_loss_kind: Some(objectness_loss_fn),
        classification_loss_kind: Some(classification_loss_fn),
        objectness_pos_weight: objectness_positive_weight,
        objectness_loss_weight: objectness_loss_weight.map(|val| val.raw()),
        classification_loss_weight: classification_loss_weight.map(|val| val.raw()),
        ..Default::default()
    }
    .build(&root / "loss")?;
    let yolo_inference = YoloInferenceInit {
        iou_threshold: r64(0.9),
        confidence_threshold: r64(0.9),
    }
    .build()?;
    let mut training_step_tensor = root.zeros_no_train("training_step", &[]);
    let mut optimizer = {
        let TrainingConfig {
            optimizer:
                OptimizerConfig {
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
                LoadCheckpoint::Disabled => 0,
                _ => f32::from(&training_step_tensor) as usize + 1,
            },
        }
    };

    // training
    {
        info!("start training");
        let mut training_step = init_training_step;
        let mut rate_counter = RateCounter::with_second_intertal();
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        let mut lr_scheduler = LrScheduler::new(lr_schedule, init_training_step)?;

        // set init lr
        {
            let init_lr = lr_scheduler.next();
            optimizer.set_lr(init_lr);
        }

        loop {
            let mut record = runtime.block_on(data_rx.recv())?;
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
            timing.add_event("forward");

            // compute loss
            let (losses, loss_auxiliary) = yolo_loss.forward(&output, &bboxes);
            timing.add_event("loss");

            // optimizer
            optimizer.backward_step(&losses.total_loss);
            timing.add_event("backward");

            // run inference
            let inference = config.logging.enable_inference.then(|| {
                let inference = yolo_inference.forward(&output);
                inference
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

            // send to logger
            logging_tx
                .send(LoggingMessage::new_training_output(
                    "training-output",
                    TrainingOutputLog {
                        step: training_step,
                        lr: r64(lr_scheduler.lr()),
                        input: image,
                        output,
                        losses,
                        matchings: loss_auxiliary.matchings,
                        inference,
                    },
                ))
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
