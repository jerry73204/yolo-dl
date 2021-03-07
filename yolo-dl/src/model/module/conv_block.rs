use super::*;

#[derive(Debug, Clone)]
pub struct ConvBlockInit {
    pub in_c: usize,
    pub out_c: usize,
    pub k: usize,
    pub s: usize,
    pub g: usize,
    pub with_activation: bool,
}

impl ConvBlockInit {
    pub fn new(in_c: usize, out_c: usize) -> Self {
        Self {
            in_c,
            out_c,
            k: 1,
            s: 1,
            g: 1,
            with_activation: true,
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
            k,
            s,
            g,
            with_activation,
        } = self;

        let conv = nn::conv2d(
            path / "conv",
            in_c as i64,
            out_c as i64,
            k as i64,
            nn::ConvConfig {
                stride: s as i64,
                padding: k as i64 / 2,
                groups: g as i64,
                bias: false,
                ..Default::default()
            },
        );
        let bn = nn::batch_norm2d(path / "bn", out_c as i64, Default::default());

        Box::new(move |xs, train| {
            let xs = xs.apply(&conv).apply_t(&bn, train);
            if with_activation {
                xs.leaky_relu()
            } else {
                xs
            }
        })
    }
}
