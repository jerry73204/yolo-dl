use super::*;

#[derive(Debug, Clone)]
pub struct SppCsp2DInit {
    pub in_c: usize,
    pub out_c: usize,
    pub k: Vec<usize>,
    pub c_mul: R64,
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
        } = self;

        let mid_c = (in_c as f64 * c_mul.raw()).floor() as usize;
        let first_conv = ConvBn2DInit::new(in_c, mid_c, 1).build(path);
        let last_conv = ConvBn2DInit::new(mid_c * 2, out_c, 1).build(path);
        let skip_conv = ConvBn2DInit::new(mid_c, mid_c, 1).build(path);

        let spp_conv_1 = ConvBn2DInit::new(mid_c, mid_c, 1).build(path);
        let spp_conv_2 = ConvBn2DInit::new(mid_c, mid_c, 3).build(path);
        let spp_conv_3 = ConvBn2DInit::new(mid_c, mid_c, 1).build(path);
        let spp_conv_4 = ConvBn2DInit::new(mid_c, mid_c, 1).build(path);
        let spp_conv_5 = ConvBn2DInit::new(mid_c, mid_c, 3).build(path);

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
                let spp = iter.fold(first, |acc, xs| acc + xs);

                spp
            };
            let xs = spp_conv_4.forward_t(&spp, train);
            let xs = spp_conv_5.forward_t(&xs, train);
            xs
        };

        let merge = Tensor::cat(&[skip, spp], 1);
        let last = last_conv.forward_t(&merge, train);
        last
    }
}
