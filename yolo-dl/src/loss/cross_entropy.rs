use crate::common::*;

#[derive(Debug)]
pub struct CrossEntropyLoss {
    reduction: Reduction,
    sparse_target: bool,
}

impl CrossEntropyLoss {
    pub fn new(sparse_target: bool, reduction: Reduction) -> Self {
        Self {
            reduction,
            sparse_target,
        }
    }

    pub fn forward(&self, input: &Tensor, target: &Tensor) -> Tensor {
        // assume [batch_size, n_classes] input shape
        let (batch_size, num_classes) = input.size2().unwrap();

        let target = if self.sparse_target {
            debug_assert!(
                target.kind() == Kind::Int64 && target.size1().unwrap() == batch_size,
                "expect target a [{}] int64 tensor",
                batch_size
            );
            debug_assert!(
                bool::from(target.ge(0).all()) && bool::from(target.lt(num_classes).all()),
                "target values must be in range of [0, {}]",
                num_classes
            );

            target.shallow_clone()
        } else {
            debug_assert!(target.size2().unwrap() == (batch_size, num_classes));
            let (_, sparse_target) = target.max2(1, false);
            sparse_target.set_requires_grad(false)
        };

        // return zero tensor if (1) input is empty and (2) using mean reduction
        if input.is_empty() && self.reduction == Reduction::Mean {
            return Tensor::zeros(&[], (Kind::Float, input.device())).set_requires_grad(false);
        }

        let loss = input.cross_entropy_for_logits(&target);

        match self.reduction {
            Reduction::None => loss,
            Reduction::Sum => loss.sum(Kind::Float),
            Reduction::Mean => loss.mean(Kind::Float),
            Reduction::Other(_) => unimplemented!(),
        }
    }
}
