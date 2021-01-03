use crate::common::*;

pub use model_config::config::Activation;

pub trait TensorEx {
    fn activation(&self, act: Activation) -> Tensor;
}

impl TensorEx for Tensor {
    fn activation(&self, act: Activation) -> Tensor {
        use Activation::*;

        match act {
            Mish => self.mish(),
            HardMish => self.hard_mish(),
            Swish => self.swish(),
            Relu => self.relu(),
            _ => unimplemented!(),
        }
    }
}
