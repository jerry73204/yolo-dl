use super::{
    conv_nd::{Conv1DInit, ConvND, ConvNDInit, ConvNDInitDyn, ConvParam},
    dark_batch_norm::{DarkBatchNorm, DarkBatchNormConfig},
};
use crate::{activation::Activation, common::*, tensor::TensorExt};

#[derive(Debug, Clone)]
pub struct ConvBnNDInit<Param: ConvParam> {
    pub conv: ConvNDInit<Param>,
    pub bn: DarkBatchNormConfig,
    pub bn_first: bool,
    pub activation: Activation,
}

pub type ConvBn1DInit = ConvBnNDInit<usize>;
pub type ConvBn2DInit = ConvBnNDInit<[usize; 2]>;
pub type ConvBn3DInit = ConvBnNDInit<[usize; 3]>;
pub type ConvBn4DInit = ConvBnNDInit<[usize; 4]>;
pub type ConvBnNDInitDyn = ConvBnNDInit<Vec<usize>>;

impl ConvBn1DInit {
    pub fn new(ksize: usize) -> Self {
        Self {
            conv: Conv1DInit::new(ksize),
            bn: Default::default(),
            bn_first: true,
            activation: Activation::Logistic,
        }
    }
}

impl<const DIM: usize> ConvBnNDInit<[usize; DIM]> {
    pub fn new(ksize: usize) -> Self {
        Self {
            conv: ConvNDInit::<[usize; DIM]>::new(ksize),
            bn: Default::default(),
            bn_first: true,
            activation: Activation::Logistic,
        }
    }
}

impl ConvBnNDInitDyn {
    pub fn new(ndim: usize, ksize: usize) -> Self {
        Self {
            conv: ConvNDInitDyn::new(ndim, ksize),
            bn: Default::default(),
            bn_first: true,
            activation: Activation::Logistic,
        }
    }
}

impl<Param: ConvParam> ConvBnNDInit<Param> {
    pub fn build<'a>(
        self,
        path: impl Borrow<nn::Path<'a>>,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<ConvBnND> {
        let path = path.borrow();
        let Self {
            conv,
            bn,
            bn_first,
            activation,
        } = self;
        let nd = conv.dim()?;

        Ok(ConvBnND {
            conv: conv.build(path / "conv", in_dim, out_dim)?,
            bn: DarkBatchNorm::new(
                path / "bn",
                nd,
                if bn_first { in_dim } else { out_dim } as i64,
                bn,
            ),
            bn_first,
            activation,
        })
    }
}

#[derive(Debug)]
pub struct ConvBnND {
    conv: ConvND,
    bn: DarkBatchNorm,
    bn_first: bool,
    activation: Activation,
}

impl ConvBnND {
    pub fn forward_t(&mut self, input: &Tensor, train: bool) -> Result<Tensor> {
        let Self {
            ref conv,
            ref mut bn,
            bn_first,
            activation,
        } = *self;

        let output = if bn_first {
            let xs = bn.forward_t(input, train)?;
            let xs = conv.forward(&xs);
            let xs = xs.activation(activation);
            xs
        } else {
            let xs = conv.forward(input);
            let xs = xs.activation(activation);
            let xs = bn.forward_t(&xs, train)?;
            xs
        };

        Ok(output)
    }
}
