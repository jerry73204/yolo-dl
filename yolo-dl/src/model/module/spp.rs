use super::*;

pub struct SppInit {
    pub in_c: usize,
    pub out_c: usize,
    pub ks: Vec<usize>,
}

impl SppInit {
    pub fn new(in_c: usize, out_c: usize) -> Self {
        Self {
            in_c,
            out_c,
            ks: vec![5, 9, 13],
        }
    }

    pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor, bool) -> Tensor + Send>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();

        let Self { in_c, out_c, ks } = self;
        let intermediate_channels = in_c / 2;

        let conv1 = ConvBlockInit {
            k: 1,
            s: 1,
            ..ConvBlockInit::new(in_c, intermediate_channels)
        }
        .build(path / "conv1");

        let conv2 = ConvBlockInit {
            k: 1,
            s: 1,
            ..ConvBlockInit::new(intermediate_channels * (ks.len() + 1), out_c)
        }
        .build(path / "conv2");

        Box::new(move |xs, train| {
            let transformed_xs = conv1(xs, train);

            let pyramid_iter = ks.iter().cloned().map(|k| {
                let k = k as i64;
                let padding = k / 2;
                let s = 1;
                let dilation = 1;
                let ceil_mode = false;
                transformed_xs.max_pool2d(
                    &[k, k],
                    &[s, s],
                    &[padding, padding],
                    &[dilation, dilation],
                    ceil_mode,
                )
            });
            let cat_xs = Tensor::cat(
                &iter::once(transformed_xs.shallow_clone())
                    .chain(pyramid_iter)
                    .collect::<Vec<_>>(),
                1,
            );

            conv2(&cat_xs, train)
        })
    }
}
