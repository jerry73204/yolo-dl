use crate::{
    common::*,
    conv_bn_2d::{ConvBn2D, ConvBn2DGrad, ConvBn2DInit},
    dark_batch_norm::DarkBatchNormInit,
};

#[derive(Debug, Clone)]
pub struct SppCsp2DInit {
    pub in_c: usize,
    pub out_c: usize,
    pub k: Vec<usize>,
    pub c_mul: R64,
    pub batch_norm: Option<DarkBatchNormInit>,
}

impl SppCsp2DInit {
    pub fn build<'p, P>(self, path: P) -> SppCsp2D
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();
        let Self {
            in_c,
            out_c,
            k,
            c_mul,
            batch_norm,
        } = self;

        let mid_c = (in_c as f64 * c_mul.raw()).floor() as usize;
        let first_conv = ConvBn2DInit {
            batch_norm: batch_norm.clone(),
            ..ConvBn2DInit::new(in_c, mid_c, 1)
        }
        .build(path / "first_conv");
        let last_conv = ConvBn2DInit {
            batch_norm: batch_norm.clone(),
            ..ConvBn2DInit::new(mid_c * 2, out_c, 1)
        }
        .build(path / "last_conv");
        let skip_conv = ConvBn2DInit {
            batch_norm: batch_norm.clone(),
            ..ConvBn2DInit::new(mid_c, mid_c, 1)
        }
        .build(path / "skip_conv");

        let spp_conv_1 = ConvBn2DInit {
            batch_norm: batch_norm.clone(),
            ..ConvBn2DInit::new(mid_c, mid_c, 1)
        }
        .build(path / "spp_conv_1");
        let spp_conv_2 = ConvBn2DInit {
            batch_norm: batch_norm.clone(),
            ..ConvBn2DInit::new(mid_c, mid_c, 3)
        }
        .build(path / "spp_conv_2");
        let spp_conv_3 = ConvBn2DInit {
            batch_norm: batch_norm.clone(),
            ..ConvBn2DInit::new(mid_c, mid_c, 1)
        }
        .build(path / "spp_conv_3");
        let spp_conv_4 = ConvBn2DInit {
            batch_norm: batch_norm.clone(),
            ..ConvBn2DInit::new(mid_c, mid_c, 1)
        }
        .build(path / "spp_conv_4");
        let spp_conv_5 = ConvBn2DInit {
            batch_norm,
            ..ConvBn2DInit::new(mid_c, mid_c, 3)
        }
        .build(path / "spp_conv_5");

        SppCsp2D {
            first_conv,
            last_conv,
            skip_conv,
            spp_conv_1,
            spp_conv_2,
            spp_conv_3,
            spp_conv_4,
            spp_conv_5,
            k,
        }
    }
}

#[derive(Debug)]
pub struct SppCsp2D {
    first_conv: ConvBn2D,
    last_conv: ConvBn2D,
    skip_conv: ConvBn2D,
    spp_conv_1: ConvBn2D,
    spp_conv_2: ConvBn2D,
    spp_conv_3: ConvBn2D,
    spp_conv_4: ConvBn2D,
    spp_conv_5: ConvBn2D,
    k: Vec<usize>,
}

impl SppCsp2D {
    pub fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let SppCsp2D {
            first_conv,
            last_conv,
            skip_conv,
            spp_conv_1,
            spp_conv_2,
            spp_conv_3,
            spp_conv_4,
            spp_conv_5,
            k,
        } = self;

        let first = first_conv.forward_t(xs, train);
        let skip = skip_conv.forward_t(&first, train);

        let spp: Tensor = {
            let xs = spp_conv_1.forward_t(&first, train);
            let xs = spp_conv_2.forward_t(&xs, train);
            let xs = spp_conv_3.forward_t(&xs, train);
            let spp: Tensor = {
                let mut iter = k.iter().cloned().map(|k| {
                    let k = k as i64;
                    let p = k / 2;
                    let s = 1;
                    let d = 1;
                    let ceil_mode = false;
                    xs.max_pool2d(&[k, k], &[s, s], &[p, p], &[d, d], ceil_mode)
                });
                let first = iter.next().unwrap();
                iter.fold(first, |acc, xs| acc + xs)
            };
            let xs = spp_conv_4.forward_t(&spp, train);
            spp_conv_5.forward_t(&xs, train)
        };

        let merge = Tensor::cat(&[skip, spp], 1);

        last_conv.forward_t(&merge, train)
    }

    pub fn grad(&self) -> SppCsp2DGrad {
        let Self {
            first_conv,
            last_conv,
            skip_conv,
            spp_conv_1,
            spp_conv_2,
            spp_conv_3,
            spp_conv_4,
            spp_conv_5,
            ..
        } = self;

        SppCsp2DGrad {
            first_conv: first_conv.grad(),
            last_conv: last_conv.grad(),
            skip_conv: skip_conv.grad(),
            spp_conv_1: spp_conv_1.grad(),
            spp_conv_2: spp_conv_2.grad(),
            spp_conv_3: spp_conv_3.grad(),
            spp_conv_4: spp_conv_4.grad(),
            spp_conv_5: spp_conv_5.grad(),
        }
    }

    pub fn clamp_bn_var(&mut self) {
        let Self {
            first_conv,
            last_conv,
            skip_conv,
            spp_conv_1,
            spp_conv_2,
            spp_conv_3,
            spp_conv_4,
            spp_conv_5,
            ..
        } = self;

        first_conv.clamp_bn_var();
        last_conv.clamp_bn_var();
        skip_conv.clamp_bn_var();
        spp_conv_1.clamp_bn_var();
        spp_conv_2.clamp_bn_var();
        spp_conv_3.clamp_bn_var();
        spp_conv_4.clamp_bn_var();
        spp_conv_5.clamp_bn_var();
    }

    pub fn denormalize_bn(&mut self) {
        let Self {
            first_conv,
            last_conv,
            skip_conv,
            spp_conv_1,
            spp_conv_2,
            spp_conv_3,
            spp_conv_4,
            spp_conv_5,
            ..
        } = self;

        first_conv.denormalize_bn();
        last_conv.denormalize_bn();
        skip_conv.denormalize_bn();
        spp_conv_1.denormalize_bn();
        spp_conv_2.denormalize_bn();
        spp_conv_3.denormalize_bn();
        spp_conv_4.denormalize_bn();
        spp_conv_5.denormalize_bn();
    }
}

#[derive(Debug, TensorLike)]
pub struct SppCsp2DGrad {
    pub first_conv: ConvBn2DGrad,
    pub last_conv: ConvBn2DGrad,
    pub skip_conv: ConvBn2DGrad,
    pub spp_conv_1: ConvBn2DGrad,
    pub spp_conv_2: ConvBn2DGrad,
    pub spp_conv_3: ConvBn2DGrad,
    pub spp_conv_4: ConvBn2DGrad,
    pub spp_conv_5: ConvBn2DGrad,
}
