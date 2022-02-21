use crate::Activation;
use tch::{nn, Tensor};

impl nn::Module for Activation {
    fn forward(&self, xs: &Tensor) -> Tensor {
        use Activation::*;

        match *self {
            Linear => xs.shallow_clone(),
            Mish => xs.mish(),
            HardMish => hard_mish(xs),
            Swish => swish(xs),
            Relu => xs.relu(),
            Leaky => leaky(xs),
            Logistic => xs.sigmoid(),
            LRelu => lrelu(xs),
            Elu => xs.elu(),
            Selu => xs.selu(),
            Gelu => xs.gelu(),
            Tanh => xs.tanh(),
            Hardtan => xs.hardtanh(),
            _ => todo!(),
        }
    }
}

pub fn leaky(xs: &Tensor) -> Tensor {
    xs.clamp_min(0.0) + xs.clamp_max(0.0) * 0.1
}

pub fn hard_mish(tensor: &Tensor) -> Tensor {
    let case1 = tensor.clamp(-2.0, 0.0);
    let case2 = tensor.clamp_min(0.0);
    (case1.pow(&2i64.into()) / 2.0 + &case1) + case2
}

pub fn swish(tensor: &Tensor) -> Tensor {
    tensor * tensor.sigmoid()
}

pub fn lrelu(tensor: &Tensor) -> Tensor {
    leaky_relu_ext(tensor, Some(0.2))
}

pub fn leaky_relu_ext(tensor: &Tensor, negative_slope: Option<f64>) -> Tensor {
    tensor.maximum(&(tensor * negative_slope.unwrap_or(0.01)))
}
