use crate::common::*;
use tch_goodies::TensorExt as _;

/// Focal loss initializer.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct FocalLossInit<F>
where
    F: 'static + Fn(&Tensor, &Tensor) -> Tensor + Send,
{
    /// The wrapped loss function.
    #[derivative(Debug = "ignore")]
    pub loss_fn: F,
    /// The gamma coefficient.
    pub gamma: f64,
    /// The alpha coefficient.
    pub alpha: f64,
    /// The reduction method applied on output loss.
    pub reduction: Reduction,
}

impl<F> FocalLossInit<F>
where
    F: 'static + Fn(&Tensor, &Tensor) -> Tensor + Send,
{
    pub fn default(reduction: Reduction, loss_fn: F) -> Self {
        Self {
            loss_fn,
            gamma: 1.5,
            alpha: 0.25,
            reduction,
        }
    }

    /// Build a focal loss calculator.
    pub fn build(self) -> FocalLoss {
        let Self {
            loss_fn,
            gamma,
            alpha,
            reduction,
        } = self;

        FocalLoss {
            loss_fn: Box::new(loss_fn),
            gamma,
            alpha,
            reduction,
        }
    }
}

/// Focal loss calculator.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct FocalLoss {
    #[derivative(Debug = "ignore")]
    loss_fn: Box<dyn Fn(&Tensor, &Tensor) -> Tensor + Send>,
    gamma: f64,
    alpha: f64,
    reduction: Reduction,
}

impl FocalLoss {
    /// Compute focal loss from an input against to a ground truth.
    pub fn forward(&self, input: &Tensor, target: &Tensor) -> Tensor {
        debug_assert_eq!(
            input.size2().unwrap(),
            target.size2().unwrap(),
            "input and target shape must be equal"
        );
        debug_assert!(
            bool::from(target.ge(0.0).logical_and(&target.le(1.0)).all()),
            "target values must be in range of [0.0, 1.0]"
        );

        // return zero tensor if (1) input is empty and (2) using mean reduction
        if input.is_empty() && self.reduction == Reduction::Mean {
            return Tensor::zeros(&[], (Kind::Float, input.device())).set_requires_grad(false);
        }

        let Self {
            ref loss_fn,
            gamma,
            alpha,
            reduction,
        } = *self;

        let orig_loss = loss_fn(input, target);
        debug_assert_eq!(
            orig_loss.size2().unwrap(),
            target.size2().unwrap(),
            "the contained loss function must not apply reduction"
        );

        let input_prob = input.sigmoid();
        let p_t: Tensor = target * &input_prob + (1.0 - target) * (1.0 - &input_prob);
        let alpha_factor = target * alpha + (1.0 - target) * (1.0 - alpha);
        let modulating_factor = (-&p_t + 1.0).pow(&gamma.into());
        let loss: Tensor = &orig_loss * &alpha_factor * &modulating_factor;

        match reduction {
            Reduction::None => loss,
            Reduction::Sum => loss.sum(Kind::Float),
            Reduction::Mean => loss.mean(Kind::Float),
            Reduction::Other(_) => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bce_with_logits_loss::BceWithLogitsLossInit;
    use rand::prelude::*;

    #[test]
    fn focal_loss() -> Result<()> {
        let mut rng = rand::thread_rng();
        let device = Device::Cpu;

        let n_batch = 32;
        let n_class = rng.gen_range(1..10);

        let vs = nn::VarStore::new(device);
        let root = vs.root();
        let loss_fn = {
            let bce = BceWithLogitsLossInit::default(Reduction::None).build(&root / "loss");
            let focal = FocalLossInit::default(Reduction::Mean, move |input, target| {
                bce.forward(input, target)
            })
            .build();
            focal
        };

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
