//! Data logging toolkit.

use crate::{
    common::*,
    config::{Config, LoggingConfig},
    utils::{CowTensor, RateCounter},
};
use async_std::{fs::File, io::BufWriter};

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
                target_bboxes,
                inference,
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

                        // gather data from each bbox
                        // btlbr = batch + tlbr
                        let (batch_indexes_vec, flat_indexes_vec, target_btlbrs) = target_bboxes
                            .0
                            .iter()
                            .map(|(pred_index, target_bbox)| {
                                let InstanceIndex { batch_index, .. } = *pred_index;
                                let flat_index = output.instance_to_flat_index(pred_index).unwrap();
                                let tlbr: PixelTLBR<f64> = target_bbox
                                    .cycxhw
                                    .scale_to_unit(image_h as f64, image_w as f64)
                                    .unwrap()
                                    .into();

                                let target_t = (tlbr.t() as i64).max(0).min(image_h - 1);
                                let target_b = (tlbr.b() as i64).max(0).min(image_h - 1);
                                let target_l = (tlbr.l() as i64).max(0).min(image_w - 1);
                                let target_r = (tlbr.r() as i64).max(0).min(image_w - 1);

                                let target_btlbr =
                                    [batch_index as i64, target_t, target_l, target_b, target_r];

                                (batch_index as i64, flat_index, target_btlbr)
                            })
                            .unzip_n_vec();

                        // draw predicted bboxes
                        {
                            let batch_indexes = Tensor::of_slice(&batch_indexes_vec);
                            let flat_indexes = Tensor::of_slice(&flat_indexes_vec);

                            let pred_cy =
                                output
                                    .cy
                                    .index_opt((&batch_indexes, NONE_INDEX, &flat_indexes));
                            let pred_cx =
                                output
                                    .cx
                                    .index_opt((&batch_indexes, NONE_INDEX, &flat_indexes));
                            let pred_h =
                                output
                                    .h
                                    .index_opt((&batch_indexes, NONE_INDEX, &flat_indexes));
                            let pred_w =
                                output
                                    .w
                                    .index_opt((&batch_indexes, NONE_INDEX, &flat_indexes));

                            let pred_t: Vec<f64> = (&pred_cy - &pred_h / 2.0).into();
                            let pred_b: Vec<f64> = (&pred_cy + &pred_h / 2.0).into();
                            let pred_l: Vec<f64> = (&pred_cx - &pred_w / 2.0).into();
                            let pred_r: Vec<f64> = (&pred_cx + &pred_w / 2.0).into();

                            let pred_btlbrs =
                                izip!(batch_indexes_vec, pred_t, pred_l, pred_b, pred_r)
                                    .map(|args| {
                                        let (batch_index, t, l, b, r) = args;
                                        let t =
                                            ((t * image_h as f64) as i64).max(0).min(image_h - 1);
                                        let b =
                                            ((b * image_h as f64) as i64).max(0).min(image_h - 1);
                                        let l =
                                            ((l * image_w as f64) as i64).max(0).min(image_w - 1);
                                        let r =
                                            ((r * image_w as f64) as i64).max(0).min(image_w - 1);
                                        [batch_index as i64, t, l, b, r]
                                    })
                                    .collect_vec();

                            // TODO: select color according to classification
                            // let pred_classification =
                            //     output.classification().index_select(0, &flat_indexes);
                            // let pred_objectness =
                            //     output.objectness().index_select(0, &flat_indexes);
                            let _ = canvas.batch_draw_rect_(&pred_btlbrs, 1, &pred_color);
                        }

                        // draw target bboxes
                        let _ = canvas.batch_draw_rect_(&target_btlbrs, 2, &target_color);

                        timing.add_event("draw bboxes");
                        Some(canvas)
                    } else {
                        None
                    };

                    // plot objectness image
                    let objectness_image = if enable_debug_stat && enable_images {
                        let batch_size = output.batch_size();
                        let objectness_maps: Vec<_> = output
                            .feature_maps()
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

                            // gather data from each bbox
                            // btlbr = batch + tlbr
                            let target_btlbrs: Vec<_> = target_bboxes
                                .0
                                .iter()
                                .map(|(pred_index, target_bbox)| {
                                    let InstanceIndex { batch_index, .. } = *pred_index;
                                    let tlbr: PixelTLBR<f64> = target_bbox
                                        .cycxhw
                                        .scale_to_unit(image_h as f64, image_w as f64)
                                        .unwrap()
                                        .into();

                                    let target_t = (tlbr.t() as i64).max(0).min(image_h - 1);
                                    let target_b = (tlbr.b() as i64).max(0).min(image_h - 1);
                                    let target_l = (tlbr.l() as i64).max(0).min(image_w - 1);
                                    let target_r = (tlbr.r() as i64).max(0).min(image_w - 1);

                                    let target_btlbr = [
                                        batch_index as i64,
                                        target_t,
                                        target_l,
                                        target_b,
                                        target_r,
                                    ];

                                    target_btlbr
                                })
                                .collect();

                            // draw predicted bboxes
                            {
                                let YoloInferenceOutput { bbox, batches, .. } = inference;
                                let batch_indexes = Vec::<i64>::from(&batches);
                                let pred_t: Vec<f64> = bbox.t().into();
                                let pred_l: Vec<f64> = bbox.l().into();
                                let pred_b: Vec<f64> = bbox.b().into();
                                let pred_r: Vec<f64> = bbox.r().into();

                                let pred_btlbrs =
                                    izip!(batch_indexes, pred_t, pred_l, pred_b, pred_r)
                                        .map(|args| {
                                            let (batch_index, t, l, b, r) = args;
                                            let t = ((t * image_h as f64) as i64)
                                                .max(0)
                                                .min(image_h - 1);
                                            let b = ((b * image_h as f64) as i64)
                                                .max(0)
                                                .min(image_h - 1);
                                            let l = ((l * image_w as f64) as i64)
                                                .max(0)
                                                .min(image_w - 1);
                                            let r = ((r * image_w as f64) as i64)
                                                .max(0)
                                                .min(image_w - 1);
                                            [batch_index as i64, t, l, b, r]
                                        })
                                        .collect_vec();

                                // TODO: select color according to classification
                                // let pred_classification =
                                //     output.classification().index_select(0, &flat_indexes);
                                // let pred_objectness =
                                //     output.objectness().index_select(0, &flat_indexes);
                                let _ = canvas.batch_draw_rect_(&pred_btlbrs, 1, &pred_color);
                            }

                            // draw target bboxes
                            let _ = canvas.batch_draw_rect_(&target_btlbrs, 2, &target_color);

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
        pub output: MergeDetect2DOutput,
        pub losses: YoloLossOutput,
        #[tensor_like(clone)]
        pub target_bboxes: PredTargetMatching,
        pub inference: Option<YoloInferenceOutput>,
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
