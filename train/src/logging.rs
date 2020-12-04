use crate::{
    common::*,
    config::{Config, LoggingConfig},
    message::{LoggingMessage, LoggingMessageKind},
    util::RateCounter,
};
use async_std::{fs::File, io::BufWriter};

struct LoggingWorker {
    config: Arc<Config>,
    debug_step: i64,
    event_writer: EventWriter<BufWriter<File>>,
    rate_counter: RateCounter,
    rx: broadcast::Receiver<LoggingMessage>,
}

impl LoggingWorker {
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

        async_std::fs::create_dir_all(&event_dir).await?;

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

    async fn start(mut self) -> Result<()> {
        loop {
            let LoggingMessage { tag, kind } = match self.rx.recv().await {
                Ok(msg) => msg,
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => break,
            };
            self.rate_counter.add(1.0);

            match kind {
                LoggingMessageKind::TrainingStep { step, losses } => {
                    self.log_training_step(&tag, step, losses).await?;
                }
                LoggingMessageKind::TrainingOutput {
                    step,
                    input,
                    output,
                    losses,
                    target_bboxes,
                } => {
                    self.log_training_output(&tag, step, input, output, losses, target_bboxes)
                        .await?;
                }
                LoggingMessageKind::Images { images } => {
                    self.log_images(&tag, images).await?;
                }
                LoggingMessageKind::ImagesWithBBoxes { tuples } => {
                    self.log_images_with_bboxes(&tag, tuples).await?;
                }
            }

            if let Some(rate) = self.rate_counter.rate() {
                info!("processed {:.2} events/s", rate);
            }
        }

        Ok(())
    }

    async fn log_training_step(
        &mut self,
        tag: &str,
        step: usize,
        losses: YoloLossOutput,
    ) -> Result<()> {
        let step = step as i64;
        self.event_writer
            .write_scalar_async(format!("{}-loss", tag), step, losses.total_loss.into())
            .await?;
        self.event_writer
            .write_scalar_async(format!("{}-iou_loss", tag), step, losses.iou_loss.into())
            .await?;
        self.event_writer
            .write_scalar_async(
                format!("{}-classification_loss", tag),
                step,
                losses.classification_loss.into(),
            )
            .await?;
        self.event_writer
            .write_scalar_async(
                format!("{}-objectness_loss", tag),
                step,
                losses.objectness_loss.into(),
            )
            .await?;
        Ok(())
    }

    async fn log_training_output(
        &mut self,
        tag: &str,
        step: usize,
        input: Tensor,
        output: YoloOutput,
        losses: YoloLossOutput,
        target_bboxes: Arc<HashMap<Arc<InstanceIndex>, Arc<LabeledGridBBox<R64>>>>,
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

        let mut timing = Timing::new("log training output");
        let step = step as i64;

        // compute statistics and plot image
        let (losses, debug_stat, bbox_image, mut timing) =
            async_std::task::spawn_blocking(move || -> Result<_> {
                tch::no_grad(|| -> Result<_> {
                    // log statistics
                    let debug_stat = if enable_debug_stat {
                        let cy_mean = f32::from(output.cy().mean(Kind::Float));
                        let cx_mean = f32::from(output.cx().mean(Kind::Float));
                        let h_mean = f32::from(output.height().mean(Kind::Float));
                        let w_mean = f32::from(output.width().mean(Kind::Float));
                        Some((cy_mean, cx_mean, h_mean, w_mean))
                    } else {
                        None
                    };

                    timing.set_record("compute_statistics");

                    // plot bboxes
                    let bbox_image = if enable_images {
                        let mut canvas = input.copy();
                        let layer_meta = output.layer_meta();
                        let PixelSize {
                            height: image_height,
                            width: image_width,
                            ..
                        } = *output.image_size();
                        let target_color = Tensor::of_slice(&[1.0, 1.0, 0.0]);
                        let pred_color = Tensor::of_slice(&[0.0, 1.0, 0.0]);

                        // gather data from each bbox
                        // btlbr = batch + tlbr
                        let (pred_info, batch_indexes_vec, flat_indexes_vec, target_btlbrs) =
                            target_bboxes
                                .iter()
                                .map(|(pred_index, target_in_grids)| -> Result<_> {
                                    let InstanceIndex {
                                        batch_index,
                                        layer_index,
                                        ..
                                    } = *pred_index.as_ref();
                                    let LabeledGridBBox {
                                        bbox: target_bbox, ..
                                    } = target_in_grids.as_ref();
                                    let flat_index =
                                        output.instance_to_flat_index(pred_index.as_ref());
                                    let [target_t, target_l, target_b, target_r] =
                                        target_bbox.tlbr();
                                    let LayerMeta {
                                        grid_size:
                                            PixelSize {
                                                height: grid_height,
                                                width: grid_width,
                                                ..
                                            },
                                        ..
                                    } = layer_meta[layer_index];

                                    let target_t = ((target_t * grid_height).raw() as i64)
                                        .max(0)
                                        .min(image_height - 1);
                                    let target_b = ((target_b * grid_height).raw() as i64)
                                        .max(0)
                                        .min(image_height - 1);
                                    let target_l = ((target_l * grid_width).raw() as i64)
                                        .max(0)
                                        .min(image_width - 1);
                                    let target_r = ((target_r * grid_width).raw() as i64)
                                        .max(0)
                                        .min(image_width - 1);

                                    let target_btlbr = [
                                        batch_index as i64,
                                        target_t,
                                        target_l,
                                        target_b,
                                        target_r,
                                    ];

                                    Ok((
                                        (batch_index, grid_height, grid_width),
                                        batch_index as i64,
                                        flat_index,
                                        target_btlbr,
                                    ))
                                })
                                .collect::<Fallible<Vec<_>>>()?
                                .into_iter()
                                .unzip_n_vec();

                        // draw predicted bboxes
                        {
                            let batch_indexes = Tensor::of_slice(&batch_indexes_vec);
                            let flat_indexes = Tensor::of_slice(&flat_indexes_vec);

                            let pred_cy = output
                                .cy()
                                .permute(&[0, 2, 1])
                                .index(&[&batch_indexes, &flat_indexes]);
                            let pred_cx = output
                                .cx()
                                .permute(&[0, 2, 1])
                                .index(&[&batch_indexes, &flat_indexes]);
                            let pred_h = output
                                .height()
                                .permute(&[0, 2, 1])
                                .index(&[&batch_indexes, &flat_indexes]);
                            let pred_w = output
                                .width()
                                .permute(&[0, 2, 1])
                                .index(&[&batch_indexes, &flat_indexes]);

                            let pred_t: Vec<f64> = (&pred_cy - &pred_h / 2.0).into();
                            let pred_b: Vec<f64> = (&pred_cy + &pred_h / 2.0).into();
                            let pred_l: Vec<f64> = (&pred_cx - &pred_w / 2.0).into();
                            let pred_r: Vec<f64> = (&pred_cx + &pred_w / 2.0).into();

                            let pred_btlbrs = izip!(pred_info, pred_t, pred_l, pred_b, pred_r)
                                .map(|args| {
                                    let ((batch_index, grid_height, grid_width), t, l, b, r) = args;
                                    let t = ((t * grid_height.raw()) as i64)
                                        .max(0)
                                        .min(image_height - 1);
                                    let b = ((b * grid_height.raw()) as i64)
                                        .max(0)
                                        .min(image_height - 1);
                                    let l =
                                        ((l * grid_width.raw()) as i64).max(0).min(image_width - 1);
                                    let r =
                                        ((r * grid_width.raw()) as i64).max(0).min(image_width - 1);
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

                        timing.set_record("draw bboxes");
                        Some(canvas)
                    } else {
                        None
                    };

                    // let objectness_image = if enable_images {
                    //     // TODO

                    //     Some(())
                    // } else {
                    //     None
                    // };

                    Ok((losses, debug_stat, bbox_image, timing))
                })
            })
            .await?;

        // log losses
        self.event_writer
            .write_scalar_async(
                format!("{}/loss/total-loss", tag),
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
        if let Some(bbox_image) = bbox_image {
            self.event_writer
                .write_image_list_async(format!("{}/image/bboxes", tag), step, bbox_image)
                .await?;

            timing.set_record("write events");
        }

        timing.report();
        Ok(())
    }

    async fn log_images(&mut self, tag: &str, images: Vec<Tensor>) -> Result<()> {
        let Config {
            logging: LoggingConfig { enable_images, .. },
            ..
        } = *self.config;

        if enable_images {
            for (index, image) in images.into_iter().enumerate() {
                self.event_writer
                    .write_image_async(format!("{}/{}", tag, index), self.debug_step, image)
                    .await?;
            }
            self.debug_step += 1;
        }

        Ok(())
    }

    async fn log_images_with_bboxes(
        &mut self,
        tag: &str,
        tuples: Vec<(Tensor, Vec<LabeledRatioBBox>)>,
    ) -> Result<()> {
        let Config {
            logging: LoggingConfig { enable_images, .. },
            ..
        } = *self.config;

        if enable_images {
            let color = Tensor::of_slice(&[1.0, 1.0, 0.0]);

            let image_vec: Vec<_> = tuples
                .into_iter()
                .map(|(image, bboxes)| {
                    let (_n_channels, height, width) = match image.size().as_slice() {
                        &[_bsize, n_channels, _height, _width] => (n_channels, _height, _width),
                        &[n_channels, _height, _width] => (n_channels, _height, _width),
                        _ => bail!("invalid shape: expec three or four dims"),
                    };
                    let mut canvas = image.copy();
                    for labeled_bbox in bboxes {
                        let LabeledRatioBBox { bbox, .. } = labeled_bbox;
                        let [cy, cx, h, w] = bbox.cycxhw();
                        let cy = cy.to_f64();
                        let cx = cx.to_f64();
                        let h = h.to_f64();
                        let w = w.to_f64();

                        let top = cy - h / 2.0;
                        let left = cx - w / 2.0;
                        let bottom = top + h;
                        let right = left + w;

                        let top = (top * height as f64) as i64;
                        let left = (left * width as f64) as i64;
                        let bottom = (bottom * height as f64) as i64;
                        let right = (right * width as f64) as i64;

                        tch::no_grad(|| {
                            let _ = canvas.draw_rect_(top, left, bottom, right, 1, &color);
                        });
                    }

                    Ok(canvas)
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
    Ok(async_std::task::spawn(worker.start()))
}
