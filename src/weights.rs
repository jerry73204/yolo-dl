use crate::common::*;

// weight types

#[derive(Debug, Clone)]
pub enum Weights {
    Connected(ConnectedWeights),
    Convolutional(ConvolutionalWeights),
    BatchNorm(BatchNormWeights),
    Shortcut(ShortcutWeights),
}

impl From<ConnectedWeights> for Weights {
    fn from(weights: ConnectedWeights) -> Self {
        Self::Connected(weights)
    }
}

impl From<ConvolutionalWeights> for Weights {
    fn from(weights: ConvolutionalWeights) -> Self {
        Self::Convolutional(weights)
    }
}

impl From<BatchNormWeights> for Weights {
    fn from(weights: BatchNormWeights) -> Self {
        Self::BatchNorm(weights)
    }
}

impl From<ShortcutWeights> for Weights {
    fn from(weights: ShortcutWeights) -> Self {
        Self::Shortcut(weights)
    }
}

#[derive(Debug, Clone)]
pub struct ScaleWeights {
    pub scales: Vec<f32>,
    pub rolling_mean: Vec<f32>,
    pub rolling_variance: Vec<f32>,
}

impl ScaleWeights {
    pub fn new(size: u64) -> Self {
        let size = size as usize;
        Self {
            scales: vec![0.0; size],
            rolling_mean: vec![0.0; size],
            rolling_variance: vec![0.0; size],
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConnectedWeights {
    pub biases: Vec<f32>,
    pub weights: Vec<f32>,
    pub scales: Option<ScaleWeights>,
}

#[derive(Debug, Clone)]
pub enum ConvolutionalWeights {
    Owned {
        biases: Vec<f32>,
        weights: Vec<f32>,
        scales: Option<ScaleWeights>,
    },
    Ref {
        share_index: usize,
    },
}

#[derive(Debug, Clone)]
pub struct BatchNormWeights {
    pub biases: Vec<f32>,
    pub scales: Vec<f32>,
    pub rolling_mean: Vec<f32>,
    pub rolling_variance: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ShortcutWeights {
    pub weights: Option<Vec<f32>>,
}
