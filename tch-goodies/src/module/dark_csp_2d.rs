use super::{
    conv_bn_2d::{ConvBn2D, ConvBn2DGrad, ConvBn2DInit},
    dark_batch_norm::DarkBatchNormInit,
};
use crate::common::*;

#[derive(Debug, Clone)]
pub struct DarkCsp2DInit {
    pub in_c: usize,
    pub out_c: usize,
    pub repeat: usize,
    pub shortcut: bool,
    pub c_mul: R64,
    pub batch_norm: Option<DarkBatchNormInit>,
}

impl DarkCsp2DInit {
    pub fn build<'p, P>(self, path: P) -> DarkCsp2D
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();
        let Self {
            in_c,
            out_c,
            repeat,
            shortcut,
            c_mul,
            batch_norm,
        } = self;

        let mid_c = (in_c as f64 * c_mul.raw()).floor() as usize;

        let skip_conv = ConvBn2DInit {
            batch_norm: batch_norm.clone(),
            ..ConvBn2DInit::new(in_c, mid_c, 1)
        }
        .build(path / "skip_conv");
        let merge_conv = ConvBn2DInit {
            batch_norm: batch_norm.clone(),
            ..ConvBn2DInit::new(mid_c * 2, out_c, 1)
        }
        .build(path / "merge_conv");
        let before_repeat_conv = ConvBn2DInit {
            batch_norm: batch_norm.clone(),
            ..ConvBn2DInit::new(in_c, mid_c, 1)
        }
        .build(path / "before_repeat_conv");
        let after_repeat_conv = ConvBn2DInit {
            batch_norm: batch_norm.clone(),
            ..ConvBn2DInit::new(mid_c, mid_c, 1)
        }
        .build(path / "after_repeat_conv");

        let repeat_convs: Vec<_> = (0..repeat)
            .map(|_| {
                let first_conv = ConvBn2DInit {
                    batch_norm: batch_norm.clone(),
                    ..ConvBn2DInit::new(mid_c, mid_c, 1)
                }
                .build(path / "first_conv");
                let second_conv = ConvBn2DInit {
                    batch_norm: batch_norm.clone(),
                    ..ConvBn2DInit::new(mid_c, mid_c, 3)
                }
                .build(path / "second_conv");
                (first_conv, second_conv)
            })
            .collect();

        DarkCsp2D {
            skip_conv,
            merge_conv,
            before_repeat_conv,
            after_repeat_conv,
            repeat_convs,
            shortcut,
        }
    }
}

#[derive(Debug)]
pub struct DarkCsp2D {
    skip_conv: ConvBn2D,
    merge_conv: ConvBn2D,
    before_repeat_conv: ConvBn2D,
    after_repeat_conv: ConvBn2D,
    repeat_convs: Vec<(ConvBn2D, ConvBn2D)>,
    shortcut: bool,
}

impl DarkCsp2D {
    pub fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let Self {
            ref skip_conv,
            ref merge_conv,
            ref before_repeat_conv,
            ref after_repeat_conv,
            ref repeat_convs,
            shortcut,
        } = *self;

        let skip = skip_conv.forward_t(xs, train);
        let repeat = {
            let xs = before_repeat_conv.forward_t(xs, train);
            let xs = repeat_convs
                .iter()
                .fold(xs, |xs, (first_conv, second_conv)| {
                    let ys = second_conv.forward_t(&first_conv.forward_t(&xs, train), train);
                    if shortcut {
                        xs + ys
                    } else {
                        ys
                    }
                });
            after_repeat_conv.forward_t(&xs, train)
        };
        let merge = Tensor::cat(&[skip, repeat], 1);
        merge_conv.forward_t(&merge, train)
    }

    pub fn grad(&self) -> DarkCsp2DGrad {
        let Self {
            skip_conv,
            merge_conv,
            before_repeat_conv,
            after_repeat_conv,
            repeat_convs,
            ..
        } = self;

        DarkCsp2DGrad {
            skip_conv: skip_conv.grad(),
            merge_conv: merge_conv.grad(),
            before_repeat_conv: before_repeat_conv.grad(),
            after_repeat_conv: after_repeat_conv.grad(),
            repeat_convs: repeat_convs
                .iter()
                .map(|(first, second)| (first.grad(), second.grad()))
                .collect(),
        }
    }

    pub fn clamp_bn_var(&mut self) {
        let Self {
            skip_conv,
            merge_conv,
            before_repeat_conv,
            after_repeat_conv,
            repeat_convs,
            ..
        } = self;

        skip_conv.clamp_bn_var();
        merge_conv.clamp_bn_var();
        before_repeat_conv.clamp_bn_var();
        after_repeat_conv.clamp_bn_var();
        repeat_convs.iter_mut().for_each(|(first, second)| {
            first.clamp_bn_var();
            second.clamp_bn_var();
        });
    }

    pub fn denormalize_bn(&mut self) {
        let Self {
            skip_conv,
            merge_conv,
            before_repeat_conv,
            after_repeat_conv,
            repeat_convs,
            ..
        } = self;

        skip_conv.denormalize_bn();
        merge_conv.denormalize_bn();
        before_repeat_conv.denormalize_bn();
        after_repeat_conv.denormalize_bn();
        repeat_convs.iter_mut().for_each(|(first, second)| {
            first.denormalize_bn();
            second.denormalize_bn();
        });
    }
}

#[derive(Debug, TensorLike)]
pub struct DarkCsp2DGrad {
    pub skip_conv: ConvBn2DGrad,
    pub merge_conv: ConvBn2DGrad,
    pub before_repeat_conv: ConvBn2DGrad,
    pub after_repeat_conv: ConvBn2DGrad,
    pub repeat_convs: Vec<(ConvBn2DGrad, ConvBn2DGrad)>,
}
