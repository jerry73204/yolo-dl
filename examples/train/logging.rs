use crate::{common::*, config::Config, message::LoggingMessage};

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

        loop {
            let msg = match rx.recv().await {
                Ok(msg) => msg,
                Err(broadcast::RecvError::Lagged(_)) => continue,
                Err(broadcast::RecvError::Closed) => break,
            };

            match msg {
                LoggingMessage::Images { tag, images } => {
                    for (index, image) in images.into_iter().enumerate() {
                        event_writer
                            .write_image_async(format!("{}/{}", tag, index), debug_step, image)
                            .await?;
                    }

                    debug_step += 1;
                }
            }
        }

        Fallible::Ok(())
    };
    let future = async_std::task::spawn(future);

    Ok(future)
}
