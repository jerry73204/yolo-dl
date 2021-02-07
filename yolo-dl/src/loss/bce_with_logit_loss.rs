use crate::common::*;

#[derive(Debug)]
pub struct BceWithLogitsLossInit {
    pub weight: Option<Tensor>,
    pub pos_weight: Option<Tensor>,
    pub reduction: Reduction,
}

impl BceWithLogitsLossInit {
    pub fn default(reduction: Reduction) -> Self {
        Self {
            weight: None,
            pos_weight: None,
            reduction,
        }
    }

    pub fn build(self) -> BceWithLogitsLoss {
        let Self {
            weight,
            pos_weight,
            reduction,
        } = self;

        BceWithLogitsLoss {
            weight,
            pos_weight,
            reduction,
        }
    }
}

#[derive(Debug)]
pub struct BceWithLogitsLoss {
    weight: Option<Tensor>,
    pos_weight: Option<Tensor>,
    reduction: Reduction,
}

impl BceWithLogitsLoss {
    pub fn forward(&self, input: &Tensor, target: &Tensor) -> Tensor {
        // assume [batch_size, n_classes] shape
        debug_assert_eq!(
            input.size2().unwrap(),
            target.size2().unwrap(),
            "input and target tensors must have equal shape"
        );
        debug_assert!(
            bool::from(target.ge(0.0).logical_and(&target.le(1.0)).all()),
            "target values must be in range of [0.0, 1.0]"
        );

        // return zero tensor if (1) input is empty and (2) using mean reduction
        if input.is_empty() && self.reduction == Reduction::Mean {
            return Tensor::zeros(&[], (Kind::Float, input.device())).set_requires_grad(false);
        }

        input.binary_cross_entropy_with_logits(
            target,
            self.weight.as_ref(),
            self.pos_weight.as_ref(),
            self.reduction,
        )
    }
}
