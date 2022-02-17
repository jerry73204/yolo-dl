use crate::common::*;
use tch_goodies::TensorExt as _;

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
            let (_, sparse_target) = target.max_dim(1, false);
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn cross_entropy_loss() -> Result<()> {
        let mut rng = rand::thread_rng();
        let device = Device::Cpu;

        let n_batch = 32;
        let n_class = rng.gen_range(1..10);

        let vs = nn::VarStore::new(device);
        let root = vs.root();
        let loss_fn = CrossEntropyLoss::new(true, Reduction::Mean);

        let input = root.randn("input", &[n_batch, n_class], 0.0, 100.0);
        let target =
            Tensor::randint(n_class, &[n_batch], (Kind::Int64, device)).set_requires_grad(false);

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

        let accuracy = i64::from(
            input
                .max_dim(1, false)
                .1
                .eq_tensor(&target)
                .count_nonzero(0),
        ) as f64
            / n_batch as f64;
        ensure!(accuracy >= 0.99, "the loss does not coverage");

        Ok(())
    }
}
