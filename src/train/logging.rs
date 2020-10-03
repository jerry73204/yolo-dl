use crate::{
    common::*,
    config::Config,
    message::{LoggingMessage, LoggingMessageKind},
    util::{RateCounter, TensorEx, Timing},
};

pub async fn logging_worker(
    config: Arc<Config>,
    mut rx: broadcast::Receiver<LoggingMessage>,
) -> Result<impl Future<Output = Result<()>> + Send> {
    let Config {
        ref logging_dir,
        log_images,
        ..
    } = *config;

    // prepare dirs
    let event_dir = logging_dir.join("events");
    let event_path_prefix = event_dir
        .join("yolo-dl-")
        .into_os_string()
        .into_string()
        .unwrap();

    async_std::fs::create_dir_all(&event_dir).await?;

    // start logging worker
    let future = async move {
        let mut debug_step = 0;
        let mut event_writer = EventWriterInit::default()
            .from_prefix_async(event_path_prefix, None)
            .await?;
        let mut rate_counter = RateCounter::with_second_intertal();
        let mut lagging_counter = RateCounter::with_second_intertal();

        loop {
            let LoggingMessage { tag, kind } = match rx.recv().await {
                Ok(msg) => {
                    rate_counter.add(1.0);
                    msg
                }
                Err(broadcast::RecvError::Lagged(_)) => {
                    lagging_counter.add(1.0);
                    continue;
                }
                Err(broadcast::RecvError::Closed) => break,
            };

            match kind {
                LoggingMessageKind::TrainingStep { step, loss } => {
                    event_writer
                        .write_scalar_async(tag, step as i64, loss)
                        .await?;
                }
                LoggingMessageKind::TrainingOutput {
                    input,
                    output,
                    losses,
                } => {
                    if log_images {
                        let mut timing = Timing::new();
                        let mut canvas = input.copy();

                        let (mut canvas, mut timing) =
                            async_std::task::spawn_blocking(move || -> Result<_> {
                                let input = input.to_device(Device::Cpu);
                                let output = output.to_device(Device::Cpu);
                                let losses = losses.to_device(Device::Cpu);

                                timing.set_record("to cpu");

                                let YoloLossOutput { target_bboxes, .. } = &losses;

                                // let layer_meta = output.layer_meta();
                                let color = Tensor::of_slice(&[1.0, 1.0, 0.0]);

                                // draw target bboxes
                                let flat_indexes: Vec<_> = target_bboxes
                                    .iter()
                                    .map(|(pred_index, target)| -> Result<_> {
                                        let InstanceIndex { batch_index, .. } =
                                            *pred_index.as_ref();
                                        let LabeledGridBBox {
                                            bbox: target_bbox,
                                            category_id,
                                        } = target.as_ref();
                                        let flat_index = output.to_flat_index(pred_index.as_ref());
                                        let [target_t, target_l, target_b, target_r] =
                                            target_bbox.tlbr();

                                        tch::no_grad(|| -> Result<_> {
                                            // TODO: select color according to category_id
                                            let _ =
                                                canvas.select(0, batch_index as i64).draw_rect_(
                                                    target_t.raw() as i64,
                                                    target_l.raw() as i64,
                                                    target_b.raw() as i64,
                                                    target_r.raw() as i64,
                                                    1,
                                                    &color,
                                                );
                                            Ok(())
                                        })?;

                                        Ok(flat_index)
                                    })
                                    .try_collect()?;

                                // draw predicted bboxes
                                {
                                    let flat_indexes = Tensor::of_slice(&flat_indexes);
                                    let pred_cy = output.cy().index_select(0, &flat_indexes);
                                    let pred_cx = output.cx().index_select(0, &flat_indexes);
                                    let pred_h = output.height().index_select(0, &flat_indexes);
                                    let pred_w = output.width().index_select(0, &flat_indexes);
                                    let pred_t = &pred_cy - &pred_h / 2.0;
                                    let pred_b = &pred_cy + &pred_h / 2.0;
                                    let pred_l = &pred_cx - &pred_w / 2.0;
                                    let pred_r = &pred_cx + &pred_w / 2.0;

                                    // TODO: select color according to classification
                                    // let pred_classification =
                                    //     output.classification().index_select(0, &flat_indexes);
                                    // let pred_objectness =
                                    //     output.objectness().index_select(0, &flat_indexes);

                                    let _ = canvas.batch_draw_rect_(
                                        &pred_t, &pred_l, &pred_b, &pred_r, 2, &color,
                                    )?;
                                }

                                timing.set_record("draw");
                                Ok((canvas, timing))
                            })
                            .await?;

                        event_writer
                            .write_image_list_async(tag, debug_step, canvas)
                            .await?;
                        debug_step += 1;

                        timing.set_record("write");
                        // info!("{:#?}", timing.records());
                    }
                }
                LoggingMessageKind::Images { images } => {
                    if log_images {
                        for (index, image) in images.into_iter().enumerate() {
                            event_writer
                                .write_image_async(format!("{}/{}", tag, index), debug_step, image)
                                .await?;
                        }
                        debug_step += 1;
                    }
                }
                LoggingMessageKind::ImagesWithBBoxes { tuples } => {
                    if log_images {
                        let color = Tensor::of_slice(&[1.0, 1.0, 0.0]);

                        let image_vec: Vec<_> = tuples
                            .into_iter()
                            .map(|(image, bboxes)| {
                                let (_n_channels, height, width) = match image.size().as_slice() {
                                    &[_bsize, n_channels, _height, _width] => {
                                        (n_channels, _height, _width)
                                    }
                                    &[n_channels, _height, _width] => (n_channels, _height, _width),
                                    _ => bail!("invalid shape: expec three or four dims"),
                                };
                                let mut canvas = image.copy();
                                for labeled_bbox in bboxes {
                                    let LabeledRatioBBox {
                                        bbox:
                                            RatioBBox {
                                                cycxhw: [cy, cx, h, w],
                                                ..
                                            },
                                        ..
                                    } = labeled_bbox;
                                    let cy = cy.raw();
                                    let cx = cx.raw();
                                    let h = h.raw();
                                    let w = w.raw();

                                    let top = cy - h / 2.0;
                                    let left = cx - w / 2.0;
                                    let bottom = top + h;
                                    let right = left + w;

                                    let top = (top * height as f64) as i64;
                                    let left = (left * width as f64) as i64;
                                    let bottom = (bottom * height as f64) as i64;
                                    let right = (right * width as f64) as i64;

                                    tch::no_grad(|| -> Result<_> {
                                        let _ = canvas
                                            .draw_rect_(top, left, bottom, right, 1, &color)?;
                                        Ok(())
                                    })?;
                                }

                                Ok(canvas)
                            })
                            .try_collect()?;
                        let images = Tensor::stack(&image_vec, 0);

                        event_writer
                            .write_image_list_async(tag, debug_step, images)
                            .await?;

                        debug_step += 1;
                    }
                }
            }

            if let Some(rate) = rate_counter.rate() {
                info!("processed {:.2} events/s", rate);
            }

            if let Some(rate) = lagging_counter.rate() {
                info!("missed {:.2} events/s", rate);
            }
        }

        Fallible::Ok(())
    };
    let future = async_std::task::spawn(future);

    Ok(future)
}
