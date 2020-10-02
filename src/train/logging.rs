use crate::{
    common::*,
    config::Config,
    message::{LoggingMessage, LoggingMessageKind},
    util::{RateCounter, TensorEx},
};

pub async fn logging_worker(
    config: Arc<Config>,
    mut rx: broadcast::Receiver<LoggingMessage>,
) -> Result<impl Future<Output = Result<()>> + Send> {
    let Config { logging_dir, .. } = &*config;

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
                Ok(msg) => msg,
                Err(broadcast::RecvError::Lagged(_)) => continue,
                Err(broadcast::RecvError::Closed) => break,
            };

            match kind {
                LoggingMessageKind::TrainingStep { step, loss } => {
                    event_writer
                        .write_scalar_async(tag, step as i64, loss)
                        .await?;
                }
                LoggingMessageKind::Images { images } => {
                    for (index, image) in images.into_iter().enumerate() {
                        event_writer
                            .write_image_async(format!("{}/{}", tag, index), debug_step, image)
                            .await?;
                    }

                    debug_step += 1;
                }
                LoggingMessageKind::ImagesWithBBoxes { tuples } => {
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

                                let _ = canvas.draw_rect_(top, left, bottom, right, 1, &color)?;
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

            rate_counter.add(1.0);
            if let Some(rate) = rate_counter.rate() {
                info!("rate {:.2} msg/s", rate);
            }
        }

        Fallible::Ok(())
    };
    let future = async_std::task::spawn(future);

    Ok(future)
}
