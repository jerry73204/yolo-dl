use super::*;

#[derive(Debug, Clone)]
pub struct FocusInit {
    pub in_c: usize,
    pub out_c: usize,
    pub k: usize,
}

impl FocusInit {
    pub fn new(in_c: usize, out_c: usize) -> Self {
        Self { in_c, out_c, k: 1 }
    }

    pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor, bool) -> Tensor + Send>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();
        let Self { in_c, out_c, k } = self;
        let conv = ConvBlockInit {
            k,
            s: 1,
            ..ConvBlockInit::new(in_c * 4, out_c)
        }
        .build(path / "conv");

        Box::new(move |xs, train| {
            let (_bsize, _channels, height, width) = xs.size4().unwrap();
            let xs = Tensor::cat(
                &[
                    xs.slice(2, 0, height, 2).slice(3, 0, width, 2),
                    xs.slice(2, 1, height, 2).slice(3, 0, width, 2),
                    xs.slice(2, 0, height, 2).slice(3, 1, width, 2),
                    xs.slice(2, 1, height, 2).slice(3, 1, width, 2),
                ],
                1,
            );
            conv(&xs, train)
        })
    }
}
