//! The model adaptor.

use crate::{common::*, config};
use model_graph as graph;

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
    pub fn new<'a>(path: impl Borrow<nn::Path<'a>>, config: &config::Model) -> Result<Self> {
        match config {
            config::Model::Darknet(config::DarknetModel { .. }) => {
                todo!();
            }
            config::Model::NewslabV1(config::NewslabV1Model { cfg_file }) => {
                let graph = graph::Graph::load_newslab_v1_json(cfg_file)?;
                let model = YoloModel::from_graph(path, &graph)?;

                Ok(Self::NewslabV1(NewslabV1Model { model }))
            }
        }
    }

    pub fn forward_t(&mut self, input: &Tensor, train: bool) -> Result<DenseDetectionTensorList> {
        match self {
            Self::Darknet(DarknetModel {}) => {
                todo!();
            }
            Self::NewslabV1(NewslabV1Model { model }) => model.forward_t(input, train),
        }
    }

    pub fn clamp_running_var(&mut self) {
        match self {
            Self::Darknet(DarknetModel {}) => {
                todo!();
            }
            Self::NewslabV1(NewslabV1Model { model }) => {
                model.clamp_running_var();
            }
        }
    }

    pub fn denormalize(&mut self) {
        match self {
            Self::Darknet(DarknetModel {}) => {
                todo!();
            }
            Self::NewslabV1(NewslabV1Model { model }) => {
                model.denormalize();
            }
        }
    }
}
