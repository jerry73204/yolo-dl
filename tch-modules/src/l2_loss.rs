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
        // return zero tensor if (1) input is empty and (2) using mean reduction
        if input.is_empty() && self.reduction == Reduction::Mean {
            return Tensor::zeros(&[], (Kind::Float, input.device())).set_requires_grad(false);
        }

        let loss = (input - target).pow(2);

        match self.reduction {
            Reduction::None => loss,
            Reduction::Sum => loss.sum(Kind::Float),
            Reduction::Mean => loss.mean(Kind::Float),
            Reduction::Other(_) => unimplemented!(),
        }
    }
}
