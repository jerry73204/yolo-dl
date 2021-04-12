use super::dark_batch_norm::{DarkBatchNorm, DarkBatchNormConfig, DarkBatchNormGrad};
use crate::{activation::Activation, common::*, tensor::TensorExt};

#[derive(Debug, Clone)]
pub struct ConvBn2DInit {
    pub in_c: usize,
    pub out_c: usize,
    pub k: usize,
    pub s: usize,
    pub p: usize,
    pub d: usize,
    pub g: usize,
    pub bias: bool,
    pub activation: Activation,
    pub batch_norm: Option<DarkBatchNormConfig>,
}

impl ConvBn2DInit {
    pub fn new(in_c: usize, out_c: usize, k: usize) -> Self {
        Self {
            in_c,
            out_c,
            k,
            s: 1,
            p: k / 2,
            d: 1,
            g: 1,
            bias: true,
            activation: Activation::Mish,
            batch_norm: Some(Default::default()),
        }
    }

    pub fn build<'p, P>(self, path: P) -> ConvBn2D
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();

        let Self {
            in_c,
            out_c,
            k,
            s,
            p,
            d,
            g,
            bias,
            activation,
            batch_norm,
        } = self;

        let conv = nn::conv2d(
            path / "conv",
            in_c as i64,
            out_c as i64,
            k as i64,
            nn::ConvConfig {
                stride: s as i64,
                padding: p as i64,
                dilation: d as i64,
                groups: g as i64,
                bias,
                ..Default::default()
            },
        );
        let bn = batch_norm.map(|config| DarkBatchNorm::new_2d(path / "bn", out_c as i64, config));

        ConvBn2D {
            conv,
            bn,
            activation,
        }
    }
}

#[derive(Debug)]
pub struct ConvBn2D {
    conv: nn::Conv2D,
    bn: Option<DarkBatchNorm>,
    activation: Activation,
}

impl ConvBn2D {
    pub fn forward_t(&mut self, xs: &Tensor, train: bool) -> Tensor {
        let Self {
            ref conv,
            ref mut bn,
            activation,
        } = *self;

        let xs = xs.apply(conv).activation(activation);

        match bn {
            Some(bn) => bn.forward_t(&xs, train).unwrap(),
            None => xs,
        }
    }

    pub fn grad(&self) -> ConvBn2DGrad {
        let Self {
            conv: nn::Conv2D { ws, bs, .. },
            bn,
            ..
        } = self;

        ConvBn2DGrad {
            conv: Conv2DGrad {
                ws: ws.grad(),
                bs: bs.as_ref().map(Tensor::grad),
            },
            bn: bn.as_ref().map(|bn| bn.grad()),
        }
    }
}

#[derive(Debug, TensorLike)]
pub struct Conv2DGrad {
    pub ws: Tensor,
    pub bs: Option<Tensor>,
}

#[derive(Debug, TensorLike)]
pub struct ConvBn2DGrad {
    pub conv: Conv2DGrad,
    pub bn: Option<DarkBatchNormGrad>,
}
