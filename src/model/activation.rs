use crate::common::*;

pub trait YoloActivation {
    fn swish(&self) -> Tensor;
    fn hard_swish(&self) -> Tensor;
    fn mish(&self) -> Tensor;
}

impl YoloActivation for Tensor {
    fn swish(&self) -> Tensor {
        self * self.sigmoid()
    }

    fn hard_swish(&self) -> Tensor {
        self * (self + 3.0).clamp(0.0, 6.0) / 6.0
    }

    fn mish(&self) -> Tensor {
        self * &self.softplus().tanh()
    }
}
