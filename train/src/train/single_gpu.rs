use crate::{
    common::*,
    config::{Config, LoadCheckpoint, LoggingConfig, LossConfig, TrainingConfig},
    data::TrainingRecord,
    message::LoggingMessage,
    model::Model,
    utils::{self, LrScheduler, RateCounter},
};

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

    const DUMMY_LR: f64 = 1.0;

    let mut vs = nn::VarStore::new(device);
    let root = vs.root();

    let mut model = Model::new(&root, model_config)?;
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
    let mut training_step_tensor = root.zeros_no_train("training_step", &[]);
    let mut optimizer = {
        let TrainingConfig {
            momentum,
            weight_decay,
            ..
        } = config.as_ref().training;
        let mut opt = nn::Adam {
            wd: weight_decay.raw(),
            ..Default::default()
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
        let mut timing = Timing::new("training loop");
        let mut rate_counter = RateCounter::with_second_intertal();
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        let mut lr_scheduler = LrScheduler::new(lr_schedule, init_training_step)?;

        // set init lr
        {
            let init_lr = lr_scheduler.next();
            optimizer.set_lr(init_lr);
        }

        loop {
            let record = runtime.block_on(data_rx.recv())?;
            timing.set_record("next record");

            let TrainingRecord {
                epoch,
                step: _record_step,
                image,
                bboxes,
            } = record.to_device(device);
            timing.set_record("to device");

            // forward pass
            let output = model.forward_t(&image, true)?;
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
                utils::save_checkpoint(
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
                        loss_auxiliary.target_bboxes,
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

            // update training step
            training_step += 1;
            training_step_tensor.copy_(&Tensor::from(training_step as f32));

            // report profiling
            {
                timing.report();
                timing = Timing::new("training loop");
            }
        }
    }
}
