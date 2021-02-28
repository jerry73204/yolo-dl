use super::*;

#[derive(Debug, Clone)]
pub struct DarkCsp2DInit {
    pub in_c: usize,
    pub out_c: usize,
    pub repeat: usize,
    pub shortcut: bool,
    pub c_mul: R64,
    pub batch_norm: Option<DarkBatchNormConfig>,
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
        .build(path);
        let merge_conv = ConvBn2DInit {
            batch_norm: batch_norm.clone(),
            ..ConvBn2DInit::new(mid_c * 2, out_c, 1)
        }
        .build(path);
        let before_repeat_conv = ConvBn2DInit {
            batch_norm: batch_norm.clone(),
            ..ConvBn2DInit::new(in_c, mid_c, 1)
        }
        .build(path);
        let after_repeat_conv = ConvBn2DInit {
            batch_norm: batch_norm.clone(),
            ..ConvBn2DInit::new(mid_c, mid_c, 1)
        }
        .build(path);

        let repeat_convs: Vec<_> = (0..repeat)
            .map(|_| {
                let first_conv = ConvBn2DInit {
                    batch_norm: batch_norm.clone(),
                    ..ConvBn2DInit::new(mid_c, mid_c, 1)
                }
                .build(path);
                let second_conv = ConvBn2DInit {
                    batch_norm: batch_norm.clone(),
                    ..ConvBn2DInit::new(mid_c, mid_c, 3)
                }
                .build(path);
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
            let xs = after_repeat_conv.forward_t(&xs, train);
            xs
        };
        let merge = Tensor::cat(&[skip, repeat], 1);
        let output = merge_conv.forward_t(&merge, train);
        output
    }
}
