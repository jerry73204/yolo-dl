use super::*;

#[derive(Debug, Clone)]
pub struct BottleneckInit {
    pub in_c: usize,
    pub out_c: usize,
    pub shortcut: bool,
    pub g: usize,
    pub expansion: R64,
}

impl BottleneckInit {
    pub fn new(in_c: usize, out_c: usize) -> Self {
        Self {
            in_c,
            out_c,
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
            shortcut,
            g,
            expansion,
        } = self;

        let intermediate_channels = (out_c as f64 * expansion.raw()) as usize;

        let conv1 = ConvBlockInit {
            k: 1,
            s: 1,
            ..ConvBlockInit::new(in_c, intermediate_channels)
        }
        .build(path / "conv1");
        let conv2 = ConvBlockInit {
            k: 3,
            s: 1,
            g,
            ..ConvBlockInit::new(intermediate_channels, out_c)
        }
        .build(path / "conv2");
        let with_add = shortcut && in_c == out_c;

        Box::new(move |xs, train| {
            let ys = conv1(xs, train);
            let ys = conv2(&ys, train);
            if with_add {
                xs + &ys
            } else {
                ys
            }
        })
    }
}
