use crate::{impls, Activation};
use tch::{nn::Module, Tensor};

pub trait TensorActivationExt {
    fn activation(&self, act: Activation) -> Tensor;
    fn lrelu(&self) -> Tensor;
    fn leaky_relu_ext<S>(&self, negative_slope: S) -> Tensor
    where
        S: Into<Option<f64>>;

    /// Hard-Mish activation function.
    fn hard_mish(&self) -> Tensor;

    /// Swish activation function.
    fn swish(&self) -> Tensor;
}

impl TensorActivationExt for Tensor {
    fn activation(&self, act: Activation) -> Tensor {
        act.forward(self)
    }

    fn lrelu(&self) -> Tensor {
        impls::lrelu(self)
    }

    fn leaky_relu_ext<S>(&self, negative_slope: S) -> Tensor
    where
        S: Into<Option<f64>>,
    {
        impls::leaky_relu_ext(self, negative_slope.into())
    }

    fn hard_mish(&self) -> Tensor {
        impls::hard_mish(self)
    }

    fn swish(&self) -> Tensor {
        impls::swish(self)
    }
}
