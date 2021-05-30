//! Data logging toolkit.

use crate::{
    common::*,
    config::{Config, LoggingConfig},
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
        config: Arc<Config>,
        debug_step: i64,
        event_writer: EventWriter<BufWriter<File>>,
        rate_counter: RateCounter,
        rx: broadcast::Receiver<LoggingMessage>,
    }

    impl LoggingWorker {
        /// Create a data logging worker.
        async fn new(
            config: Arc<Config>,
            logging_dir: Arc<PathBuf>,
            rx: broadcast::Receiver<LoggingMessage>,
        ) -> Result<Self> {
            let debug_step = 0;

            // prepare dirs
            let event_dir = logging_dir.join("events");
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
                let LoggingMessage { tag, kind } = match self.rx.recv().await {
                    Ok(msg) => msg,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                };
                self.rate_counter.add(1.0);

                match kind {
                    LoggingMessageKind::TrainingOutput(msg) => {
                        self.log_training_output(&tag, msg).await?;
                    }
                    LoggingMessageKind::DebugImages(msg) => {
                        self.log_debug_images(&tag, msg).await?;
                    }
                    LoggingMessageKind::DebugImagesWithCyCxHWes(msg) => {
                        self.log_debug_images_with_bboxes(&tag, msg).await?;
                    }
                }

                if let Some(rate) = self.rate_counter.rate() {
                    info!("processed {:.2} events/s", rate);
                }
            }

            Ok(())
        }

        async fn log_training_output(&mut self, tag: &str, msg: TrainingOutputLog) -> Result<()> {
            let mut timing = Timing::new("log_training_output");

            let Config {
                logging:
                    LoggingConfig {
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

                        timing.add_event("draw bboxes");
                        Some(canvas)
                    } else {
                        None
                    };

                    // plot objectness image
                    let objectness_image = if enable_debug_stat && enable_images {
                        let batch_size = output.batch_size();
                        let objectness_maps: Vec<_> = output
                            .tensors()
                            .tensors
                            .into_iter()
                            .map(|feature_map| feature_map.obj)
                            .zip_eq(output.info.iter())
                            .map(|(objectness_map, meta)| {
                                let num_anchors = meta.anchors.len() as i64;
                                objectness_map
                                    .copy()
                                    .resize_(&[batch_size, 1, num_anchors, image_h, image_w])
                                    .view([batch_size, num_anchors, image_h, image_w])
                            })
                            .collect();

                        // concatenate at "anchor" dimension, and find the max over that dimension
                        let (objectness_image, _argmax) =
                            Tensor::cat(&objectness_maps, 1).max2(1, true);
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
            .map(|result| Fallible::Ok(result??))
            .await?;

            // log parameters
            self.event_writer
                .write_scalar_async(
                    format!("{}/params/learning_rate", tag),
                    step,
                    lr.raw() as f32,
                )
                .await?;

            // log losses
            self.event_writer
                .write_scalar_async(
                    format!("{}/loss/total_loss", tag),
                    step,
                    losses.total_loss.into(),
                )
                .await?;
            self.event_writer
                .write_scalar_async(
                    format!("{}/loss/iou_loss", tag),
                    step,
                    losses.iou_loss.into(),
                )
                .await?;
            self.event_writer
                .write_scalar_async(
                    format!("{}/loss/classification_loss", tag),
                    step,
                    losses.classification_loss.into(),
                )
                .await?;
            self.event_writer
                .write_scalar_async(
                    format!("{}/loss/objectness_loss", tag),
                    step,
                    losses.objectness_loss.into(),
                )
                .await?;

            // log debug statistics
            if let Some((cy_mean, cx_mean, h_mean, w_mean)) = debug_stat {
                self.event_writer
                    .write_scalar_async(format!("{}/stat/cy_mean", tag), step, cy_mean)
                    .await?;
                self.event_writer
                    .write_scalar_async(format!("{}/stat/cx_mean", tag), step, cx_mean)
                    .await?;
                self.event_writer
                    .write_scalar_async(format!("{}/stat/h_mean", tag), step, h_mean)
                    .await?;
                self.event_writer
                    .write_scalar_async(format!("{}/stat/w_mean", tag), step, w_mean)
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
                        format!("{}/benchmark/objectness_accuracy", tag),
                        step,
                        obj_accuracy as f32,
                    )
                    .await?;
                self.event_writer
                    .write_scalar_async(
                        format!("{}/benchmark/objectness_precision", tag),
                        step,
                        obj_precision as f32,
                    )
                    .await?;
                self.event_writer
                    .write_scalar_async(
                        format!("{}/benchmark/objectness_recall", tag),
                        step,
                        obj_recall as f32,
                    )
                    .await?;
                self.event_writer
                    .write_scalar_async(
                        format!("{}/benchmark/classification_accuracy", tag),
                        step,
                        class_accuracy as f32,
                    )
                    .await?;
            }

            // write images
            if let Some(image) = training_bbox_image {
                self.event_writer
                    .write_image_list_async(format!("{}/image/training_bboxes", tag), step, image)
                    .await?;

                timing.add_event("write events");
            }

            if let Some(image) = inference_bbox_image {
                self.event_writer
                    .write_image_list_async(format!("{}/image/inference_bboxes", tag), step, image)
                    .await?;

                timing.add_event("write events");
            }

            if let Some(objectness_image) = objectness_image {
                self.event_writer
                    .write_image_list_async(
                        format!("{}/image/objectness", tag),
                        step,
                        objectness_image,
                    )
                    .await?;

                timing.add_event("write events");
            }

            timing.report();
            Ok(())
        }

        async fn log_debug_images(&mut self, tag: &str, msg: DebugImageLog) -> Result<()> {
            let Config {
                logging:
                    LoggingConfig {
                        enable_images,
                        enable_debug_stat,
                        ..
                    },
                ..
            } = *self.config;
            let DebugImageLog { images } = msg;

            if enable_debug_stat && enable_images {
                for (index, image) in images.into_iter().enumerate() {
                    self.event_writer
                        .write_image_async(format!("{}/{}", tag, index), self.debug_step, image)
                        .await?;
                }
                self.debug_step += 1;
            }

            Ok(())
        }

        async fn log_debug_images_with_bboxes(
            &mut self,
            tag: &str,
            msg: DebugLabeledImageLog,
        ) -> Result<()> {
            let Config {
                logging:
                    LoggingConfig {
                        enable_images,
                        enable_debug_stat,
                        ..
                    },
                ..
            } = *self.config;
            let DebugLabeledImageLog { images, bboxes } = msg;

            if enable_debug_stat && enable_images {
                let color = Tensor::of_slice(&[1.0, 1.0, 0.0]);

                let image_vec: Vec<_> = izip!(images, bboxes)
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
                                    labeled_bbox.cycxhw.cast::<f64>().unwrap().cycxhw_params();

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
                    .try_collect()?;
                let images = Tensor::stack(&image_vec, 0);

                self.event_writer
                    .write_image_list_async(tag, self.debug_step, images)
                    .await?;

                self.debug_step += 1;
            }
            Ok(())
        }
    }

    pub async fn logging_worker(
        config: Arc<Config>,
        logging_dir: Arc<PathBuf>,
        rx: broadcast::Receiver<LoggingMessage>,
    ) -> Result<impl Future<Output = Result<()>> + Send> {
        let worker = LoggingWorker::new(config, logging_dir, rx).await?;
        Ok(tokio::task::spawn(worker.start()).map(|result| Fallible::Ok(result??)))
    }
}

mod logging_message {
    use super::*;

    /// The message type that is accepted by the logging worker.
    #[derive(Debug, TensorLike, Clone)]
    pub struct LoggingMessage {
        #[tensor_like(clone)]
        pub tag: Cow<'static, str>,
        pub kind: LoggingMessageKind,
    }

    impl LoggingMessage {
        pub fn new_training_output<S>(tag: S, msg: TrainingOutputLog) -> Self
        where
            S: Into<Cow<'static, str>>,
        {
            Self {
                tag: tag.into(),
                kind: LoggingMessageKind::TrainingOutput(msg),
            }
        }

        pub fn new_debug_images<'a, S, I, T>(tag: S, images: I) -> Self
        where
            S: Into<Cow<'static, str>>,
            I: IntoIterator<Item = T>,
            T: Into<CowTensor<'a>>,
        {
            Self {
                tag: tag.into(),
                kind: LoggingMessageKind::DebugImages(DebugImageLog {
                    images: images
                        .into_iter()
                        .map(|tensor| tensor.into().into_owned())
                        .collect_vec(),
                }),
            }
        }

        pub fn new_debug_labeled_images<'a, S, I, IB, B, T>(tag: S, tuples: I) -> Self
        where
            S: Into<Cow<'static, str>>,
            I: IntoIterator<Item = (T, IB)>,
            IB: IntoIterator<Item = B>,
            B: Borrow<RatioLabel>,
            T: Into<CowTensor<'a>>,
        {
            let (images, bboxes) = tuples
                .into_iter()
                .map(|(tensor, bboxes)| {
                    (
                        tensor.into().into_owned(),
                        bboxes
                            .into_iter()
                            .map(|bbox| bbox.borrow().to_owned())
                            .collect_vec(),
                    )
                })
                .unzip_n_vec();

            Self {
                tag: tag.into(),
                kind: LoggingMessageKind::DebugImagesWithCyCxHWes(DebugLabeledImageLog {
                    images,
                    bboxes,
                }),
            }
        }
    }

    #[derive(Debug, TensorLike)]
    pub enum LoggingMessageKind {
        TrainingOutput(TrainingOutputLog),
        DebugImages(DebugImageLog),
        DebugImagesWithCyCxHWes(DebugLabeledImageLog),
    }

    impl Clone for LoggingMessageKind {
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
    }

    impl Clone for TrainingOutputLog {
        fn clone(&self) -> Self {
            self.shallow_clone()
        }
    }

    #[derive(Debug, TensorLike)]
    pub struct DebugImageLog {
        pub images: Vec<Tensor>,
    }

    impl Clone for DebugImageLog {
        fn clone(&self) -> Self {
            self.shallow_clone()
        }
    }

    #[derive(Debug, TensorLike)]
    pub struct DebugLabeledImageLog {
        pub images: Vec<Tensor>,
        #[tensor_like(clone)]
        pub bboxes: Vec<Vec<RatioLabel>>,
    }

    impl Clone for DebugLabeledImageLog {
        fn clone(&self) -> Self {
            self.shallow_clone()
        }
    }
}
