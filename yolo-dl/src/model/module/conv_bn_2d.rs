use super::*;

#[derive(Debug, Clone)]
pub struct ConvBn2DInit {
    pub in_c: usize,
    pub out_c: usize,
    pub k: usize,
    pub s: usize,
    pub p: usize,
    pub d: usize,
    pub g: usize,
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
        let bn = batch_norm.map(|config| DarkBatchNorm::new_2d(path, out_c as i64, config));

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

        let xs = match bn {
            Some(bn) => bn.forward_t(&xs, train).unwrap(),
            None => xs,
        };

        xs
    }
}
