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
    let (output_tx, output_rx) = async_channel::bounded(num_cpus::get() * 2);

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

    let inference_futs = workers.into_iter().map(|(vs, mut model)| {
        let data_rx = data_rx.clone();
        let config = config.clone();
        let device = vs.device();
        let output_tx = output_tx.clone();

        async move {
            while let Ok(record) = data_rx.recv().await {
                let (model_, images, output) = tokio::task::spawn_blocking(move || -> Result<_> {
                    let record = record?;
                    let InputRecord {
                        indexes,
                        images,
                        bboxes,
                    } = &record;
                    let images = images.to_device(device);
                    let output = model
                        .forward_t(&images, false)?
                        .merge_detect_2d()
                        .ok_or_else(|| format_err!("invalid model output type"))?;
                    Ok((model, record, output))
                })
                .await??;

                model = model_;
                output_tx.send((images, output)).await.unwrap();
            }
            Ok(())
        }
    });

    let output_fut = output_rx
        .flat_map(|(record, output)| {
            let InputRecord {
                indexes,
                images,
                bboxes,
            } = record;
            let batch_size = images.size4().unwrap().0;

            stream::iter((0..batch_size).zip_eq(indexes).zip_eq(bboxes).map(
                move |((batch_index, index), bbox)| {
                    let image = images.i((batch_index, .., .., ..));
                    (index, image, bbox)
                },
            ))
        })
        .par_for_each_blocking(None, |args| {
            move || {
                let (index, image, bbox) = args;
                // TODO
            }
        });

    futures::try_join!(
        scatter_fut.map(|_| Fallible::Ok(())),
        futures::future::try_join_all(inference_futs),
        output_fut.map(|_| Fallible::Ok(())),
    )?;

    Ok(())
}
