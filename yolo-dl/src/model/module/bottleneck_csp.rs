use super::*;

#[derive(Debug, Clone)]
pub struct BottleneckCspInit {
    pub in_c: usize,
    pub out_c: usize,
    pub repeat: usize,
    pub shortcut: bool,
    pub g: usize,
    pub expansion: R64,
}

impl BottleneckCspInit {
    pub fn new(in_c: usize, out_c: usize) -> Self {
        Self {
            in_c,
            out_c,
            repeat: 1,
            shortcut: true,
            g: 1,
            expansion: R64::new(0.5),
        }
    }

    pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor, bool) -> Tensor + Send>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();

        let Self {
            in_c,
            out_c,
            repeat,
            shortcut,
            g,
            expansion,
        } = self;
        debug_assert!(repeat > 0);

        let intermediate_channels = (out_c as f64 * expansion.raw()) as usize;

        let conv1 = ConvBlockInit {
            k: 1,
            s: 1,
            ..ConvBlockInit::new(in_c, intermediate_channels)
        }
        .build(path);
        let conv2 = nn::conv2d(
            path,
            in_c as i64,
            intermediate_channels as i64,
            1,
            nn::ConvConfig {
                stride: 1,
                bias: false,
                ..Default::default()
            },
        );
        let conv3 = nn::conv2d(
            path,
            intermediate_channels as i64,
            intermediate_channels as i64,
            1,
            nn::ConvConfig {
                stride: 1,
                bias: false,
                ..Default::default()
            },
        );
        let conv4 = ConvBlockInit {
            k: 1,
            s: 1,
            ..ConvBlockInit::new(out_c, out_c)
        }
        .build(path);
        let bn = nn::batch_norm2d(path, intermediate_channels as i64 * 2, Default::default());
        let bottlenecks = (0..repeat)
            .map(|_| {
                BottleneckInit {
                    shortcut,
                    g,
                    expansion: R64::new(1.0),
                    ..BottleneckInit::new(intermediate_channels, intermediate_channels)
                }
                .build(path)
            })
            .collect::<Vec<_>>();

        Box::new(move |xs, train| {
            let y1 = {
                let y = conv1(xs, train);
                let y = bottlenecks
                    .iter()
                    .fold(y, |input, block| block(&input, train));
                y.apply_t(&conv3, train)
            };
            let y2 = xs.apply_t(&conv2, train);
            conv4(
                &Tensor::cat(&[y1, y2], 1).apply_t(&bn, train).leaky_relu(),
                train,
            )
        })
    }
}
