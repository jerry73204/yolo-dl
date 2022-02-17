use tch_goodies::TensorExt as _;

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

    pub fn build<'a>(self, path: impl Borrow<nn::Path<'a>>) -> BceWithLogitsLoss {
        let Self {
            weight,
            pos_weight,
            reduction,
        } = self;

        let path = path.borrow();
        let weight = weight.map(|from| {
            tch::no_grad(|| {
                let mut weight = path.zeros_no_train("weight", &from.size());
                weight.copy_(&from);
                weight
            })
        });
        let pos_weight = pos_weight.map(|from| {
            tch::no_grad(|| {
                let mut weight = path.zeros_no_train("pos_weight", &from.size());
                weight.copy_(&from);
                weight
            })
        });

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

        let device = input.device();

        // return zero tensor if (1) input is empty and (2) using mean reduction
        if input.is_empty() && self.reduction == Reduction::Mean {
            return Tensor::zeros(&[], (Kind::Float, device)).set_requires_grad(false);
        }

        input.binary_cross_entropy_with_logits(
            target,
            self.weight.as_ref(),
            self.pos_weight.as_ref(),
            self.reduction,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn bce_loss() -> Result<()> {
        let mut rng = rand::thread_rng();
        let device = Device::Cpu;

        let n_batch = 32;
        let n_class = rng.gen_range(1..10);

        let vs = nn::VarStore::new(device);
        let root = vs.root();
        let loss_fn = BceWithLogitsLossInit::default(Reduction::Mean).build(&root / "loss");

        let input = root.randn("input", &[n_batch, n_class], 0.0, 100.0);
        let target = Tensor::rand(&[n_batch, n_class], (Kind::Float, device))
            .ge(0.5)
            .to_kind(Kind::Float)
            .set_requires_grad(false);

        let mut optimizer = nn::Adam::default().build(&vs, 1.0)?;

        for _ in 0..1000 {
            let loss = loss_fn.forward(&input, &target);
            optimizer.backward_step(&loss);
        }

        optimizer.set_lr(0.1);

        for _ in 0..10000 {
            let loss = loss_fn.forward(&input, &target);
            optimizer.backward_step(&loss);
        }

        optimizer.set_lr(0.01);

        for _ in 0..10000 {
            let loss = loss_fn.forward(&input, &target);
            optimizer.backward_step(&loss);
        }

        ensure!(
            bool::from((input.sigmoid() - &target).abs().le(1e-3).all()),
            "the loss does not coverage"
        );

        Ok(())
    }
}
