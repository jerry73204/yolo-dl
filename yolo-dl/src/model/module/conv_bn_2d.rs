use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConvBn2DInit {
    pub in_c: usize,
    pub out_c: usize,
    pub k: usize,
    pub s: usize,
    pub p: usize,
    pub d: usize,
    pub g: usize,
    pub activation: Activation,
    pub batch_norm: bool,
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
            activation: Activation::Mish,
            batch_norm: true,
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
            activation,
            batch_norm,
        } = self;

        let conv = nn::conv2d(
            path,
            in_c as i64,
            out_c as i64,
            k as i64,
            nn::ConvConfig {
                stride: s as i64,
                padding: p as i64,
                dilation: d as i64,
                groups: g as i64,
                bias: false,
                ..Default::default()
            },
        );
        let bn = if batch_norm {
            Some(nn::batch_norm2d(path, out_c as i64, Default::default()))
        } else {
            None
        };

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
    bn: Option<nn::BatchNorm>,
    activation: Activation,
}

impl ConvBn2D {
    pub fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let Self {
            ref conv,
            ref bn,
            activation,
        } = *self;

        let xs = xs.apply(conv).activation(activation);

        let xs = match bn {
            Some(bn) => xs.apply_t(bn, train),
            None => xs,
        };

        xs
    }
}
