use crate::common::*;

#[derive(Debug)]
pub struct YoloLossInit {
    pub pos_weight: Option<Tensor>,
    pub reduction: Reduction,
    pub focal_loss_gamma: Option<f64>,
    pub anchors_list: Vec<Vec<(usize, usize)>>,
}

impl YoloLossInit {
    pub fn build(self) -> YoloLoss {
        let Self {
            pos_weight,
            reduction,
            focal_loss_gamma,
            anchors_list,
        } = self;
        let focal_loss_gamma = focal_loss_gamma.unwrap_or(0.0);
        assert!(focal_loss_gamma >= 0.0);

        let bce_class = FocalLossInit {
            pos_weight: pos_weight.as_ref().map(|weight| weight.shallow_clone()),
            gamma: focal_loss_gamma,
            reduction,
            ..Default::default()
        }
        .build();

        let bce_objectness = FocalLossInit {
            pos_weight,
            gamma: focal_loss_gamma,
            reduction,
            ..Default::default()
        }
        .build();

        let gain = Tensor::ones(&[6], (Kind::Float, Device::Cpu));
        let offsets = Tensor::of_slice(&[1, 0, 0, 1, -1, 0, 0, -1]).view([3, 2]);
        let anchor_tensors = Tensor::arange(anchors_list.len() as i64, (Kind::Int64, Device::Cpu));

        YoloLoss {
            bce_class,
            bce_objectness,
            anchors_list,
        }
    }
}

impl YoloLossInit {
    pub fn new(anchors_list: Vec<Vec<(usize, usize)>>) -> Self {
        Self {
            pos_weight: None,
            reduction: Reduction::Mean,
            focal_loss_gamma: None,
            anchors_list,
        }
    }
}

#[derive(Debug)]
pub struct YoloLoss {
    bce_class: FocalLoss,
    bce_objectness: FocalLoss,
    anchors_list: Vec<Vec<(usize, usize)>>,
}

impl YoloLoss {
    pub fn forward<T1, T2>(&self, inputs: &[T1], targets: &[T2]) -> Tensor
    where
        T1: Borrow<Tensor>,
        T2: Borrow<Tensor>,
    {
        debug_assert_eq!(inputs.len(), targets.len());
        debug_assert_eq!(inputs.len(), self.anchors_list.len());

        izip!(
            inputs.iter(),
            targets.iter(),
            self.anchors_list.iter().cloned()
        )
        .map(|args| {
            let (input, target, anchors) = args;
            let input = input.borrow();
            let target = target.borrow();
        });
        todo!();
    }
}

#[derive(Debug)]
pub struct BceWithLogitsLossInit {
    pub weight: Option<Tensor>,
    pub pos_weight: Option<Tensor>,
    pub reduction: Reduction,
}

impl BceWithLogitsLossInit {
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

impl Default for BceWithLogitsLossInit {
    fn default() -> Self {
        Self {
            weight: None,
            pos_weight: None,
            reduction: Reduction::Mean,
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
        input.binary_cross_entropy_with_logits(
            target,
            self.weight.as_ref(),
            self.pos_weight.as_ref(),
            self.reduction,
        )
    }
}

#[derive(Debug)]
pub struct FocalLossInit {
    pub weight: Option<Tensor>,
    pub pos_weight: Option<Tensor>,
    pub gamma: f64,
    pub alpha: f64,
    pub reduction: Reduction,
}

impl FocalLossInit {
    pub fn build(self) -> FocalLoss {
        let Self {
            weight,
            pos_weight,
            gamma,
            alpha,
            reduction,
        } = self;

        let bce = BceWithLogitsLossInit {
            weight,
            pos_weight,
            reduction: Reduction::None,
        }
        .build();

        FocalLoss {
            bce,
            gamma,
            alpha,
            reduction,
        }
    }
}

impl Default for FocalLossInit {
    fn default() -> Self {
        Self {
            weight: None,
            pos_weight: None,
            gamma: 1.5,
            alpha: 0.25,
            reduction: Reduction::Mean,
        }
    }
}

#[derive(Debug)]
pub struct FocalLoss {
    bce: BceWithLogitsLoss,
    gamma: f64,
    alpha: f64,
    reduction: Reduction,
}

impl FocalLoss {
    pub fn forward(&self, input: &Tensor, target: &Tensor) -> Tensor {
        let Self {
            bce,
            gamma,
            alpha,
            reduction,
        } = self;

        let bce_loss = self.bce.forward(input, target);
        let input_prob = input.sigmoid();
        let p_t: Tensor = target * &input_prob + (1.0 - target) * (1.0 - &input_prob);
        let alpha_factor = target * self.alpha + (1.0 - target) * (1.0 - self.alpha);
        let modulating_factor = (-&p_t + 1.0).pow(self.gamma);
        let loss: Tensor = bce_loss * alpha_factor * modulating_factor;

        match self.reduction {
            Reduction::Mean => loss.mean(Kind::Float),
            Reduction::Sum => loss.sum(Kind::Float),
            Reduction::None => loss,
            Reduction::Other(_) => unreachable!(),
        }
    }
}
