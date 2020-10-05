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

        loop {
            let LoggingMessage { tag, kind } = match rx.recv().await {
                Ok(msg) => {
                    rate_counter.add(1.0);
                    msg
                }
                Err(broadcast::RecvError::Lagged(_)) => continue,
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

                        let (mut canvas, mut timing) =
                            async_std::task::spawn_blocking(move || -> Result<_> {
                                tch::no_grad(|| -> Result<_> {
                                    let input = input.to_device(Device::Cpu);
                                    let output = output.to_device(Device::Cpu);
                                    let losses = losses.to_device(Device::Cpu);

                                    timing.set_record("to cpu");

                                    let mut canvas = input.copy();
                                    let YoloLossOutput { target_bboxes, .. } = &losses;
                                    let layer_meta = output.layer_meta();
                                    let PixelSize {
                                        height: image_height,
                                        width: image_width,
                                        ..
                                    } = *output.image_size();
                                    let target_color = Tensor::of_slice(&[1.0, 1.0, 0.0]);
                                    let pred_color = Tensor::of_slice(&[0.0, 1.0, 0.0]);

                                    // gather data from each bbox
                                    let (pred_info, flat_indexes, target_btlbrs) = target_bboxes
                                        .iter()
                                        .map(|(pred_index, target)| -> Result<_> {
                                            let InstanceIndex {
                                                batch_index,
                                                layer_index,
                                                ..
                                            } = *pred_index.as_ref();
                                            let LabeledGridBBox {
                                                bbox: target_bbox, ..
                                            } = target.as_ref();
                                            let flat_index =
                                                output.to_flat_index(pred_index.as_ref());
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

                                            let target_t = ((target_t.raw() * grid_height) as i64)
                                                .max(0)
                                                .min(image_height - 1);
                                            let target_b = ((target_b.raw() * grid_height) as i64)
                                                .max(0)
                                                .min(image_height - 1);
                                            let target_l = ((target_l.raw() * grid_width) as i64)
                                                .max(0)
                                                .min(image_width - 1);
                                            let target_r = ((target_r.raw() * grid_width) as i64)
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
                                                flat_index,
                                                target_btlbr,
                                            ))
                                        })
                                        .collect::<Fallible<Vec<_>>>()?
                                        .into_iter()
                                        .unzip_n_vec();

                                    // draw target bboxes
                                    let _ =
                                        canvas.batch_draw_rect_(&target_btlbrs, 2, &target_color);

                                    // draw predicted bboxes
                                    {
                                        let flat_indexes = Tensor::of_slice(&flat_indexes);
                                        let pred_cy = output.cy().index_select(0, &flat_indexes);
                                        let pred_cx = output.cx().index_select(0, &flat_indexes);
                                        let pred_h = output.height().index_select(0, &flat_indexes);
                                        let pred_w = output.width().index_select(0, &flat_indexes);

                                        let pred_t: Vec<f64> = (&pred_cy - &pred_h / 2.0).into();
                                        let pred_b: Vec<f64> = (&pred_cy + &pred_h / 2.0).into();
                                        let pred_l: Vec<f64> = (&pred_cx - &pred_w / 2.0).into();
                                        let pred_r: Vec<f64> = (&pred_cx + &pred_w / 2.0).into();

                                        let pred_btlbrs =
                                            izip!(pred_info, pred_t, pred_l, pred_b, pred_r)
                                                .map(|args| {
                                                    let (
                                                        (batch_index, grid_height, grid_width),
                                                        t,
                                                        l,
                                                        b,
                                                        r,
                                                    ) = args;
                                                    let t = ((t * grid_height) as i64)
                                                        .max(0)
                                                        .min(image_height - 1);
                                                    let b = ((b * grid_height) as i64)
                                                        .max(0)
                                                        .min(image_height - 1);
                                                    let l = ((l * grid_width) as i64)
                                                        .max(0)
                                                        .min(image_width - 1);
                                                    let r = ((r * grid_width) as i64)
                                                        .max(0)
                                                        .min(image_width - 1);
                                                    [batch_index as i64, t, l, b, r]
                                                })
                                                .collect_vec();

                                        // TODO: select color according to classification
                                        // let pred_classification =
                                        //     output.classification().index_select(0, &flat_indexes);
                                        // let pred_objectness =
                                        //     output.objectness().index_select(0, &flat_indexes);

                                        let _ =
                                            canvas.batch_draw_rect_(&pred_btlbrs, 2, &pred_color);
                                    }

                                    timing.set_record("draw");
                                    Ok((canvas, timing))
                                })
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

                                    tch::no_grad(|| {
                                        let _ =
                                            canvas.draw_rect_(top, left, bottom, right, 1, &color);
                                    });
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
        }

        Fallible::Ok(())
    };
    let future = async_std::task::spawn(future);

    Ok(future)
}
