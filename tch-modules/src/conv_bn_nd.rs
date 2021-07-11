use crate::{
    common::*,
    conv_nd::{Conv1DInit, ConvND, ConvNDGrad, ConvNDInit, ConvNDInitDyn, ConvParam},
    dark_batch_norm::{DarkBatchNorm, DarkBatchNormGrad, DarkBatchNormInit},
};

#[derive(Debug, Clone)]
pub struct ConvBnInit<Param: ConvParam> {
    pub conv: ConvNDInit<Param>,
    pub bn: DarkBatchNormInit,
    pub bn_first: bool,
    pub activation: Activation,
}

pub type ConvBnInit1D = ConvBnInit<usize>;
pub type ConvBnInit2D = ConvBnInit<[usize; 2]>;
pub type ConvBnInit3D = ConvBnInit<[usize; 3]>;
pub type ConvBnInit4D = ConvBnInit<[usize; 4]>;
pub type ConvBnInitDyn = ConvBnInit<Vec<usize>>;

impl ConvBnInit1D {
    pub fn new(ksize: usize) -> Self {
        Self {
            conv: Conv1DInit::new(ksize),
            bn: Default::default(),
            bn_first: true,
            activation: Activation::Logistic,
        }
    }
}

impl<const DIM: usize> ConvBnInit<[usize; DIM]> {
    pub fn new(ksize: usize) -> Self {
        Self {
            conv: ConvNDInit::<[usize; DIM]>::new(ksize),
            bn: Default::default(),
            bn_first: true,
            activation: Activation::Logistic,
        }
    }
}

impl ConvBnInitDyn {
    pub fn new(ndim: usize, ksize: usize) -> Self {
        Self {
            conv: ConvNDInitDyn::new(ndim, ksize),
            bn: Default::default(),
            bn_first: true,
            activation: Activation::Logistic,
        }
    }
}

impl<Param: ConvParam> ConvBnInit<Param> {
    pub fn build<'a>(
        self,
        path: impl Borrow<nn::Path<'a>>,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<ConvBn> {
        let path = path.borrow();
        let Self {
            conv,
            bn,
            bn_first,
            activation,
        } = self;

        Ok(ConvBn {
            conv: conv.build(path / "conv", in_dim, out_dim)?,
            bn: bn.build(path / "bn", if bn_first { in_dim } else { out_dim } as i64),
            bn_first,
            activation,
        })
    }
}

#[derive(Debug)]
pub struct ConvBn {
    conv: ConvND,
    bn: DarkBatchNorm,
    bn_first: bool,
    activation: Activation,
}

impl nn::ModuleT for ConvBn {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        let Self {
            ref conv,
            ref bn,
            bn_first,
            activation,
        } = *self;

        let output = if bn_first {
            let xs = bn.forward_t(input, train);
            let xs = conv.forward(&xs);
            xs.activation(activation)
        } else {
            let xs = conv.forward(input);
            let xs = xs.activation(activation);
            bn.forward_t(&xs, train)
        };

        output
    }
}

impl ConvBn {
    pub fn grad(&self) -> ConvBnGrad {
        let Self { conv, bn, .. } = self;

        ConvBnGrad {
            conv: conv.grad(),
            bn: bn.grad(),
        }
    }

    pub fn clamp_running_var(&mut self) {
        self.bn.clamp_running_var();
    }

    pub fn denormalize(&mut self) {
        self.bn.denormalize();
    }
}

#[derive(Debug, TensorLike)]
pub struct ConvBnGrad {
    conv: ConvNDGrad,
    bn: DarkBatchNormGrad,
}
