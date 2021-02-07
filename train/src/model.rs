//! The model adaptor.

use crate::{
    common::*,
    config::{DarknetModelConfig, ModelConfig, NewslabV1ModelConfig},
};

/// The generic model adaptor.
#[derive(Debug)]
pub enum Model {
    /// Model built from Darknet configuration. It is not supported yet.
    Darknet(DarknetModel),
    /// Model built from NEWSLABv1 configuration.
    NewslabV1(NewslabV1Model),
}

/// Model built from Darknet configuration. It is not supported yet.
#[derive(Debug)]
pub struct DarknetModel {}

/// Model built from Darknet configuration.
#[derive(Debug)]
pub struct NewslabV1Model {
    model: YoloModel,
}

impl Model {
    /// Builds a model adaptor from a configuration file.
    pub fn new<'a>(path: impl Borrow<nn::Path<'a>>, config: &ModelConfig) -> Result<Self> {
        match config {
            ModelConfig::Darknet(DarknetModelConfig { .. }) => {
                todo!();
            }
            ModelConfig::NewslabV1(NewslabV1ModelConfig { cfg_file }) => {
                use model_config::{config, graph};

                let config = config::Model::load(cfg_file)?;
                let graph = graph::Graph::new(&config)?;
                let model = YoloModel::from_graph(path, &graph)?;

                Ok(Self::NewslabV1(NewslabV1Model { model }))
            }
        }
    }

    pub fn forward_t(&mut self, input: &Tensor, train: bool) -> Result<MergeDetect2DOutput> {
        match self {
            Self::Darknet(DarknetModel {}) => {
                todo!();
            }
            Self::NewslabV1(NewslabV1Model { model }) => {
                let output = model
                    .forward_t(input, train)?
                    .merge_detect_2d()
                    .ok_or_else(|| format_err!("TODO"))?;
                Ok(output)
            }
        }
    }
}
