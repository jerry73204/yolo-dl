mod common;
pub mod config;
pub mod input_stream;

use crate::{
    common::*,
    config::Config,
    input_stream::{InputRecord, InputStream},
};

pub async fn start(config: Arc<Config>) -> Result<()> {
    let num_devices = config.model.devices.len();

    // load model
    let workers: Vec<_> = stream::iter(config.model.devices.clone())
        .par_map_init_unordered(
            num_devices,
            || config.clone(),
            move |config, device| {
                move || -> Result<_> {
                    use model_config::{config, graph};

                    let vs = nn::VarStore::new(device);
                    let root = vs.root();
                    let config = config::Model::load(&config.model.cfg_file)?;
                    let graph = graph::Graph::new(&config)?;
                    let model = YoloModel::from_graph(root, &graph)?;

                    Ok((vs, model))
                }
            },
        )
        .try_collect()
        .await?;

    // load dataset
    let input_stream = InputStream::new(config.clone()).await?;

    // scatter input to workers
    let (scatter_fut, data_rx) = input_stream.stream()?.par_scatter(None);

    let worker_futs = workers.into_iter().map(|(vs, mut model)| {
        let data_rx = data_rx.clone();
        let config = config.clone();
        let device = vs.device();

        async move {
            while let Ok(record) = data_rx.recv().await {
                model = tokio::task::spawn_blocking(move || -> Result<_> {
                    let InputRecord {
                        indexes,
                        images,
                        bboxes,
                    } = record?;
                    let images = images.to_device(device);
                    let output = model.forward_t(&images, false)?;
                    Ok(model)
                })
                .await??;
            }
            Ok(())
        }
    });

    futures::try_join!(
        scatter_fut.map(|_| Fallible::Ok(())),
        futures::future::try_join_all(worker_futs)
    )?;

    Ok(())
}
