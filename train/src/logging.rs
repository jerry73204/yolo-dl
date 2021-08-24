//! Data logging toolkit.

use crate::{
    common::*,
    config,
    utils::{CowTensor, RateCounter},
};
use async_std::{fs::File, io::BufWriter};
use tch_goodies::MergedDenseDetection;

pub use logging_message::*;
pub use logging_worker::*;

mod logging_worker {
    use super::*;

    /// The data logging worker.
    #[derive(Debug)]
    pub struct LoggingWorker {
        config: ArcRef<config::Config>,
        debug_step: i64,
        event_writer: EventWriter<BufWriter<File>>,
        rate_counter: RateCounter,
        rx: broadcast::Receiver<LoggingMessage>,
    }

    impl LoggingWorker {
        /// Create a data logging worker.
        async fn new(
            config: ArcRef<config::Config>,
            logging_dir: impl AsRef<Path>,
            rx: broadcast::Receiver<LoggingMessage>,
        ) -> Result<Self> {
            let debug_step = 0;

            // prepare dirs
            let event_dir = logging_dir.as_ref().join("events");
            let event_path_prefix = event_dir
                .join("yolo-dl")
                .into_os_string()
                .into_string()
                .unwrap();

            tokio::fs::create_dir_all(&event_dir).await?;

            let event_writer = EventWriterInit::default()
                .from_prefix_async(event_path_prefix, None)
                .await?;
            let rate_counter = RateCounter::with_second_intertal();

            Ok(Self {
                config,
                debug_step,
                event_writer,
                rate_counter,
                rx,
            })
        }

        /// Start the data logging worker.
        async fn start(mut self) -> Result<()> {
            loop {
                let msg = match self.rx.recv().await {
                    Ok(msg) => msg,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                };

                match msg {
                    LoggingMessage::TrainingOutput(msg) => {
                        self.log_training_output(msg).await?;
                    }
                    LoggingMessage::DebugImages(msg) => {
                        self.log_debug_images(msg).await?;
                    }
                }

                self.rate_counter.add(1.0);
                if let Some(rate) = self.rate_counter.rate() {
                    info!("processed {:.2} events/s", rate);
                }
            }

            Ok(())
        }

        async fn log_training_output(&mut self, msg: TrainingOutputLog) -> Result<()> {
            let mut timing = Timing::new("log_training_output");

            let config::Config {
                logging:
                    config::Logging {
                        enable_images,
                        enable_debug_stat,
                        ..
                    },
                ..
            } = *self.config;
            let TrainingOutputLog {
                step,
                lr,
                input,
                output,
                losses,
                matchings,
                inference,
                benchmark,
                weights,
                gradients,
            } = msg;

            let step = step as i64;
            let (_b, _c, image_h, image_w) = input.size4()?;
            timing.add_event("initialize");

            // compute statistics and plot image
            let (
                losses,
                debug_stat,
                training_bbox_image,
                inference_bbox_image,
                objectness_image,
                mut timing,
            ) = tokio::task::spawn_blocking(move || -> Result<_> {
                tch::no_grad(|| -> Result<_> {
                    // log statistics
                    let debug_stat = if enable_debug_stat {
                        let cy_mean = f32::from(output.cy.mean(Kind::Float));
                        let cx_mean = f32::from(output.cx.mean(Kind::Float));
                        let h_mean = f32::from(output.h.mean(Kind::Float));
                        let w_mean = f32::from(output.w.mean(Kind::Float));
                        timing.add_event("compute_statistics");
                        Some((cy_mean, cx_mean, h_mean, w_mean))
                    } else {
                        None
                    };

                    // plot bboxes for training
                    let training_bbox_image = if enable_images {
                        // plotting is faster on CPU
                        let mut canvas = input.copy().to_device(Device::Cpu);

                        let target_color = Tensor::of_slice(&[1.0, 1.0, 0.0]);
                        let pred_color = Tensor::of_slice(&[0.0, 1.0, 0.0]);

                        // draw target bboxes
                        {
                            let tlbr = TLBRTensor::from(matchings.target.cycxhw());
                            let _ = canvas.batch_draw_ratio_rect_(
                                &matchings.pred_indexes.batches,
                                &tlbr.t().view([-1]),
                                &tlbr.l().view([-1]),
                                &tlbr.b().view([-1]),
                                &tlbr.r().view([-1]),
                                2,
                                &target_color,
                            );
                        }

                        // draw predicted bboxes
                        {
                            let pred = output.index_by_flats(&matchings.pred_indexes);
                            let tlbr = TLBRTensor::from(pred.cycxhw());
                            let _ = canvas.batch_draw_ratio_rect_(
                                &matchings.pred_indexes.batches,
                                &tlbr.t().view([-1]),
                                &tlbr.l().view([-1]),
                                &tlbr.b().view([-1]),
                                &tlbr.r().view([-1]),
                                1,
                                &pred_color,
                            );
                        }

                        timing.add_event("draw bboxes");
                        Some(canvas)
                    } else {
                        None
                    };

                    // plot objectness image
                    let objectness_image = if enable_debug_stat && enable_images {
                        let batch_size = output.batch_size();
                        let objectness_maps: Vec<_> = output
                            .to_tensor_list()
                            .tensors
                            .iter()
                            .zip_eq(output.info.iter())
                            .map(|(feature_map, meta)| {
                                let feature_h = meta.feature_size.h;
                                let feature_w = meta.feature_size.w;
                                let num_anchors = meta.anchors.len() as i64;
                                feature_map
                                    .obj_prob()
                                    .to_device(Device::Cpu)
                                    .view([batch_size, num_anchors, feature_h, feature_w])
                                    .resize2d_exact(image_h, image_w)
                            })
                            .try_collect()?;

                        // concatenate at "anchor" dimension, and find the max over that dimension
                        let (objectness_image, _argmax) =
                            Tensor::cat(&objectness_maps, 1).max_dim(1, true);
                        debug_assert!(
                            objectness_image.size4().unwrap() == (batch_size, 1, image_h, image_w)
                        );

                        timing.add_event("plot_objectness_image");
                        Some(objectness_image)
                    } else {
                        None
                    };

                    // plot bboxes from inference output
                    let inference_bbox_image = match (inference, enable_images) {
                        (Some(inference), true) => {
                            // plotting is faster on CPU
                            let mut canvas = input.copy().to_device(Device::Cpu);

                            let target_color = Tensor::of_slice(&[1.0, 1.0, 0.0]);
                            let pred_color = Tensor::of_slice(&[0.0, 1.0, 0.0]);

                            // draw target bboxes
                            {
                                let tlbr = TLBRTensor::from(matchings.target.cycxhw());
                                let _ = canvas.batch_draw_ratio_rect_(
                                    &matchings.pred_indexes.batches,
                                    &tlbr.t().view([-1]),
                                    &tlbr.l().view([-1]),
                                    &tlbr.b().view([-1]),
                                    &tlbr.r().view([-1]),
                                    1,
                                    &target_color,
                                );
                            }

                            // draw predicted bboxes
                            {
                                let YoloInferenceOutput { bbox, batches, .. } = inference;

                                // TODO: select color according to classification
                                let _ = canvas.batch_draw_ratio_rect_(
                                    &batches,
                                    &bbox.t().view([-1]),
                                    &bbox.l().view([-1]),
                                    &bbox.b().view([-1]),
                                    &bbox.r().view([-1]),
                                    1,
                                    &pred_color,
                                );
                            }

                            timing.add_event("draw inference bboxes");
                            Some(canvas)
                        }
                        _ => None,
                    };

                    Ok((
                        losses,
                        debug_stat,
                        training_bbox_image,
                        inference_bbox_image,
                        objectness_image,
                        timing,
                    ))
                })
            })
            .await??;

            // log parameters
            self.event_writer
                .write_scalar_async(format!("params/learning_rate",), step, lr.raw() as f32)
                .await?;

            // log losses
            self.event_writer
                .write_scalar_async(format!("loss/total_loss",), step, losses.total_loss.into())
                .await?;
            self.event_writer
                .write_scalar_async(format!("loss/iou_loss",), step, losses.iou_loss.into())
                .await?;
            self.event_writer
                .write_scalar_async(
                    format!("loss/classification_loss",),
                    step,
                    losses.classification_loss.into(),
                )
                .await?;
            self.event_writer
                .write_scalar_async(
                    format!("loss/objectness_loss",),
                    step,
                    losses.objectness_loss.into(),
                )
                .await?;

            // log debug statistics
            if let Some((cy_mean, cx_mean, h_mean, w_mean)) = debug_stat {
                self.event_writer
                    .write_scalar_async(format!("stat/cy_mean",), step, cy_mean)
                    .await?;
                self.event_writer
                    .write_scalar_async(format!("stat/cx_mean",), step, cx_mean)
                    .await?;
                self.event_writer
                    .write_scalar_async(format!("stat/h_mean",), step, h_mean)
                    .await?;
                self.event_writer
                    .write_scalar_async(format!("stat/w_mean",), step, w_mean)
                    .await?;
            }

            // log benchmark
            if let Some(benchmark) = benchmark {
                let YoloBenchmarkOutput {
                    obj_accuracy,
                    obj_recall,
                    obj_precision,
                    class_accuracy,
                } = benchmark;

                self.event_writer
                    .write_scalar_async(
                        format!("benchmark/objectness_accuracy",),
                        step,
                        obj_accuracy as f32,
                    )
                    .await?;
                self.event_writer
                    .write_scalar_async(
                        format!("benchmark/objectness_precision",),
                        step,
                        obj_precision as f32,
                    )
                    .await?;
                self.event_writer
                    .write_scalar_async(
                        format!("benchmark/objectness_recall",),
                        step,
                        obj_recall as f32,
                    )
                    .await?;
                self.event_writer
                    .write_scalar_async(
                        format!("benchmark/classification_accuracy",),
                        step,
                        class_accuracy as f32,
                    )
                    .await?;
            }

            // log weights and gradients
            if let Some(weights) = weights {
                for (name, weight) in weights {
                    self.event_writer
                        .write_scalar_async(format!("weights/{}", name), step, weight as f32)
                        .await?;
                }
            }

            if let Some(gradients) = gradients {
                for (name, grad) in gradients {
                    self.event_writer
                        .write_scalar_async(format!("gradients/{}", name), step, grad as f32)
                        .await?;
                }
            }

            // write images
            if let Some(image) = training_bbox_image {
                self.event_writer
                    .write_image_list_async(format!("training_bboxes",), step, image)
                    .await?;

                timing.add_event("write events");
            }

            if let Some(image) = inference_bbox_image {
                self.event_writer
                    .write_image_list_async(format!("inference_bboxes",), step, image)
                    .await?;

                timing.add_event("write events");
            }

            if let Some(objectness_image) = objectness_image {
                self.event_writer
                    .write_image_list_async(format!("objectness",), step, objectness_image)
                    .await?;

                timing.add_event("write events");
            }

            timing.report();
            Ok(())
        }

        async fn log_debug_images(&mut self, msg: DebugImageLog) -> Result<()> {
            let config::Config {
                logging:
                    config::Logging {
                        enable_images,
                        enable_debug_stat,
                        ..
                    },
                ..
            } = *self.config;
            let DebugImageLog {
                name,
                images,
                bboxes,
            } = msg;

            if enable_debug_stat && enable_images {
                let color = Tensor::of_slice(&[1.0, 1.0, 0.0]);

                let image_vec: Vec<_> = if let Some(bboxes) = bboxes {
                    ensure!(images.len() == bboxes.len());
                    izip!(images, bboxes)
                        .map(|(image, bboxes)| {
                            tch::no_grad(|| {
                                let (_c, height, width) = match image.size().as_slice() {
                                    &[_b, c, h, w] => (c, h, w),
                                    &[c, h, w] => (c, h, w),
                                    _ => bail!("invalid shape: expec three or four dims"),
                                };
                                let mut canvas = image.copy().to_device(Device::Cpu);

                                for labeled_bbox in bboxes {
                                    let [cy, cx, h, w] =
                                        labeled_bbox.rect.cast::<f64>().unwrap().cycxhw();

                                    let top = cy - h / 2.0;
                                    let left = cx - w / 2.0;
                                    let bottom = top + h;
                                    let right = left + w;

                                    let top = (top * height as f64) as i64;
                                    let left = (left * width as f64) as i64;
                                    let bottom = (bottom * height as f64) as i64;
                                    let right = (right * width as f64) as i64;

                                    let _ = canvas.draw_rect_(top, left, bottom, right, 1, &color);
                                }

                                Ok(canvas)
                            })
                        })
                        .try_collect()?
                } else {
                    images
                        .into_iter()
                        .map(|image| tch::no_grad(|| image.copy().to_device(Device::Cpu)))
                        .collect()
                };

                let images = Tensor::stack(&image_vec, 0);

                self.event_writer
                    .write_image_list_async(name, self.debug_step, images)
                    .await?;

                self.debug_step += 1;
            }
            Ok(())
        }
    }

    pub async fn logging_worker(
        config: ArcRef<config::Config>,
        logging_dir: impl AsRef<Path>,
        rx: broadcast::Receiver<LoggingMessage>,
    ) -> Result<()> {
        let worker = LoggingWorker::new(config, logging_dir, rx).await?;
        tokio::task::spawn(worker.start()).await??;
        Ok(())
    }
}

mod logging_message {
    use super::*;

    /// The message type that is accepted by the logging worker.
    #[derive(Debug, TensorLike)]
    pub enum LoggingMessage {
        TrainingOutput(TrainingOutputLog),
        DebugImages(DebugImageLog),
    }

    impl LoggingMessage {
        pub fn new_debug_images<'a, 'b>(
            name: impl Into<Cow<'a, str>>,
            images: impl IntoIterator<Item = impl Into<CowTensor<'b>>>,
            bboxes: Option<
                impl IntoIterator<Item = impl IntoIterator<Item = impl Borrow<RatioRectLabel<R64>>>>,
            >,
        ) -> Self {
            let name = name.into().into_owned();

            let images: Vec<_> = images
                .into_iter()
                .map(|image| image.into().into_owned())
                .collect();

            let bboxes: Option<Vec<_>> = bboxes.map(|bboxes| {
                bboxes
                    .into_iter()
                    .map(|bboxes| {
                        bboxes
                            .into_iter()
                            .map(|bbox| bbox.borrow().to_owned())
                            .collect_vec()
                    })
                    .collect()
            });

            Self::DebugImages(DebugImageLog {
                name,
                images,
                bboxes,
            })
        }
    }

    impl From<TrainingOutputLog> for LoggingMessage {
        fn from(v: TrainingOutputLog) -> Self {
            Self::TrainingOutput(v)
        }
    }

    impl From<DebugImageLog> for LoggingMessage {
        fn from(v: DebugImageLog) -> Self {
            Self::DebugImages(v)
        }
    }

    impl Clone for LoggingMessage {
        fn clone(&self) -> Self {
            self.shallow_clone()
        }
    }

    #[derive(Debug, TensorLike)]
    pub struct TrainingOutputLog {
        pub step: usize,
        #[tensor_like(clone)]
        pub lr: R64,
        pub input: Tensor,
        pub output: MergedDenseDetection,
        pub losses: YoloLossOutput,
        pub matchings: MatchingOutput,
        pub inference: Option<YoloInferenceOutput>,
        pub benchmark: Option<YoloBenchmarkOutput>,
        #[tensor_like(clone)]
        pub weights: Option<Vec<(String, f64)>>,
        #[tensor_like(clone)]
        pub gradients: Option<Vec<(String, f64)>>,
    }

    impl Clone for TrainingOutputLog {
        fn clone(&self) -> Self {
            self.shallow_clone()
        }
    }

    #[derive(Debug, TensorLike)]
    pub struct DebugImageLog {
        #[tensor_like(clone)]
        pub name: String,
        pub images: Vec<Tensor>,
        #[tensor_like(clone)]
        pub bboxes: Option<Vec<Vec<RatioRectLabel<R64>>>>,
    }

    impl Clone for DebugImageLog {
        fn clone(&self) -> Self {
            self.shallow_clone()
        }
    }
}
