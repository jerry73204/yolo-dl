use crate::{
    common::*,
    dark_batch_norm::{DarkBatchNorm, DarkBatchNormGrad, DarkBatchNormInit},
};

#[derive(Debug, Clone)]
pub struct DeconvBn2DInit {
    pub in_c: usize,
    pub out_c: usize,
    pub k: usize,
    pub s: usize,
    pub p: usize,
    pub op: usize,
    pub d: usize,
    pub g: usize,
    pub bias: bool,
    pub activation: Activation,
    pub batch_norm: Option<DarkBatchNormInit>,
}

impl DeconvBn2DInit {
    pub fn new(in_c: usize, out_c: usize, k: usize) -> Self {
        Self {
            in_c,
            out_c,
            k,
            s: 1,
            p: k / 2,
            op: 0,
            d: 1,
            g: 1,
            bias: true,
            activation: Activation::Mish,
            batch_norm: Some(Default::default()),
        }
    }

    pub fn build<'p, P>(self, path: P) -> DeconvBn2D
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
            op,
            d,
            g,
            bias,
            activation,
            batch_norm,
        } = self;

        let deconv = nn::conv_transpose2d(
            path / "deconv",
            in_c as i64,
            out_c as i64,
            k as i64,
            nn::ConvTransposeConfig {
                stride: s as i64,
                padding: p as i64,
                output_padding: op as i64,
                dilation: d as i64,
                groups: g as i64,
                bias,
                ..Default::default()
            },
        );
        let bn = batch_norm.map(|init| init.build(path / "bn", out_c as i64));

        DeconvBn2D {
            deconv,
            bn,
            activation,
        }
    }
}

#[derive(Debug)]
pub struct DeconvBn2D {
    deconv: nn::ConvTranspose2D,
    bn: Option<DarkBatchNorm>,
    activation: Activation,
}

impl DeconvBn2D {
    pub fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let Self {
            ref deconv,
            ref bn,
            activation,
        } = *self;

        let xs = xs.apply(deconv).activation(activation);

        match bn {
            Some(bn) => bn.forward_t(&xs, train).unwrap(),
            None => xs,
        }
    }

    pub fn grad(&self) -> DeconvBn2DGrad {
        let Self {
            deconv: nn::ConvTranspose2D { ws, bs, .. },
            bn,
            ..
        } = self;

        DeconvBn2DGrad {
            deconv: ConvTranspose2DGrad {
                ws: ws.grad(),
                bs: bs.as_ref().map(Tensor::grad),
            },
            bn: bn.as_ref().map(|bn| bn.grad()),
        }
    }

    pub fn clamp_bn_var(&mut self) {
        if let Some(bn) = &mut self.bn {
            bn.clamp_bn_var();
        }
    }

    pub fn denormalize_bn(&mut self) {
        if let Some(bn) = &mut self.bn {
            bn.denormalize_bn();
        }
    }
}

#[derive(Debug, TensorLike)]
pub struct ConvTranspose2DGrad {
    pub ws: Tensor,
    pub bs: Option<Tensor>,
}

#[derive(Debug, TensorLike)]
pub struct DeconvBn2DGrad {
    pub deconv: ConvTranspose2DGrad,
    pub bn: Option<DarkBatchNormGrad>,
}
