mod common;
pub mod config;
pub mod input_stream;

use crate::{
    common::*,
    config::{Config, ModelConfig},
    input_stream::{InputRecord, InputStream},
};

pub async fn start(config: Arc<Config>) -> Result<()> {
    // load dataset
    let input_stream = InputStream::new(config.clone()).await?;

    // load model
    let (vs, mut model) = {
        use model_config::{config, graph};

        let ModelConfig {
            ref cfg_file,
            device,
            ..
        } = config.model;

        let vs = nn::VarStore::new(device);
        let root = vs.root();

        let config = config::Model::load(cfg_file)?;
        let graph = graph::Graph::new(&config)?;
        let model = YoloModel::from_graph(root, &graph)?;

        (vs, model)
    };

    let mut stream = input_stream.stream()?;

    loop {
        let record = match stream.try_next().await? {
            Some(record) => record,
            None => break,
        };

        let InputRecord {
            indexes,
            images,
            bboxes,
        } = record;
        let images = images.to_device(config.model.device);

        let output = model.forward_t(&images, false)?;
    }

    Ok(())
}
