use crate::common::*;

#[derive(Debug)]
pub struct L2Loss {
    reduction: Reduction,
}

impl L2Loss {
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }

    pub fn forward(&self, input: &Tensor, target: &Tensor) -> Tensor {
        input.mse_loss(target, self.reduction)
    }
}
