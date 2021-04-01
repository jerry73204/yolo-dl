use crate::common::*;

pub use conv_init::*;
pub use conv_nd::*;
pub use conv_param::*;

mod conv_param {
    use super::*;

    pub trait ConvParam
    where
        Self: Clone,
    {
        fn dim(&self) -> usize;
        fn i64_iter(&self) -> Box<dyn Iterator<Item = i64>>;
        fn usize_iter(&self) -> Box<dyn Iterator<Item = usize>>;
    }

    impl ConvParam for usize {
        fn dim(&self) -> usize {
            1
        }

        fn i64_iter(&self) -> Box<dyn Iterator<Item = i64>> {
            Box::new(iter::once(*self as i64))
        }

        fn usize_iter(&self) -> Box<dyn Iterator<Item = usize>> {
            Box::new(iter::once(*self))
        }
    }

    impl<const DIM: usize> ConvParam for [usize; DIM] {
        fn dim(&self) -> usize {
            DIM
        }

        fn i64_iter(&self) -> Box<dyn Iterator<Item = i64>> {
            Box::new(Vec::from(*self).into_iter().map(|val| val as i64))
        }

        fn usize_iter(&self) -> Box<dyn Iterator<Item = usize>> {
            Box::new(Vec::from(*self).into_iter())
        }
    }

    impl ConvParam for Vec<usize> {
        fn dim(&self) -> usize {
            self.len()
        }

        fn i64_iter(&self) -> Box<dyn Iterator<Item = i64>> {
            Box::new(self.clone().into_iter().map(|val| val as i64))
        }

        fn usize_iter(&self) -> Box<dyn Iterator<Item = usize>> {
            Box::new(self.clone().into_iter())
        }
    }

    impl ConvParam for &[usize] {
        fn dim(&self) -> usize {
            self.len()
        }

        fn i64_iter(&self) -> Box<dyn Iterator<Item = i64>> {
            Box::new(Vec::<usize>::from(*self).into_iter().map(|val| val as i64))
        }

        fn usize_iter(&self) -> Box<dyn Iterator<Item = usize>> {
            Box::new(Vec::<usize>::from(*self).into_iter())
        }
    }
}

mod conv_init {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct ConvNDInit<Param: ConvParam> {
        pub ksize: Param,
        pub stride: Param,
        pub padding: Param,
        pub dilation: Param,
        pub groups: usize,
        pub bias: bool,
        pub transposed: bool,
        pub ws_init: nn::Init,
        pub bs_init: nn::Init,
    }

    pub type Conv1DInit = ConvNDInit<usize>;
    pub type Conv2DInit = ConvNDInit<[usize; 2]>;
    pub type Conv3DInit = ConvNDInit<[usize; 3]>;
    pub type Conv4DInit = ConvNDInit<[usize; 4]>;
    pub type ConvNDInitDyn = ConvNDInit<Vec<usize>>;

    impl Conv1DInit {
        pub fn new(ksize: usize) -> Self {
            Self {
                ksize,
                stride: 1,
                padding: ksize / 2,
                dilation: 1,
                groups: 1,
                bias: true,
                transposed: false,
                ws_init: nn::Init::KaimingUniform,
                bs_init: nn::Init::Const(0.0),
            }
        }
    }

    impl<const DIM: usize> ConvNDInit<[usize; DIM]> {
        pub fn new(ksize: usize) -> Self {
            Self {
                ksize: [ksize; DIM],
                stride: [1; DIM],
                padding: [ksize / 2; DIM],
                dilation: [1; DIM],
                groups: 1,
                bias: true,
                transposed: false,
                ws_init: nn::Init::KaimingUniform,
                bs_init: nn::Init::Const(0.0),
            }
        }
    }

    impl ConvNDInitDyn {
        pub fn new(ndims: usize, ksize: usize) -> Self {
            Self {
                ksize: vec![ksize; ndims],
                stride: vec![1; ndims],
                padding: vec![ksize / 2; ndims],
                dilation: vec![1; ndims],
                groups: 1,
                bias: true,
                transposed: false,
                ws_init: nn::Init::KaimingUniform,
                bs_init: nn::Init::Const(0.0),
            }
        }
    }

    impl<Param: ConvParam> ConvNDInit<Param> {
        pub fn into_dyn(self) -> ConvNDInitDyn {
            let Self {
                ksize,
                stride,
                padding,
                dilation,
                groups,
                bias,
                transposed,
                ws_init,
                bs_init,
            } = self;

            ConvNDInitDyn {
                ksize: ksize.usize_iter().collect(),
                stride: stride.usize_iter().collect(),
                padding: padding.usize_iter().collect(),
                dilation: dilation.usize_iter().collect(),
                groups,
                bias,
                transposed,
                ws_init,
                bs_init,
            }
        }

        pub fn dim(&self) -> Result<usize> {
            let Self {
                ksize,
                stride,
                padding,
                dilation,
                ..
            } = self;

            ensure!(
                ksize.dim() == stride.dim()
                    && ksize.dim() == padding.dim()
                    && ksize.dim() == dilation.dim(),
                "parameter dimension mismatch"
            );

            Ok(ksize.dim())
        }

        pub fn build<'a>(
            self,
            path: impl Borrow<nn::Path<'a>>,
            in_dim: usize,
            out_dim: usize,
        ) -> Result<ConvND> {
            let Self {
                ksize,
                stride,
                padding,
                dilation,
                groups,
                bias,
                transposed,
                ws_init,
                bs_init,
            } = self;

            ensure!(
                groups > 0 && in_dim % groups == 0,
                "in_dim must be multiple of group"
            );

            let path = path.borrow();
            let in_dim = in_dim as i64;
            let out_dim = out_dim as i64;
            let ksize: Vec<i64> = ksize.i64_iter().collect();
            let stride: Vec<i64> = stride.i64_iter().collect();
            let padding: Vec<i64> = padding.i64_iter().collect();
            let dilation: Vec<i64> = dilation.i64_iter().collect();
            let groups = groups as i64;

            let bs = bias.then(|| path.var("bias", &[out_dim], bs_init));
            let ws = {
                let weight_size: Vec<i64> = if transposed {
                    vec![in_dim, out_dim / groups]
                } else {
                    vec![out_dim, in_dim / groups]
                }
                .into_iter()
                .chain(ksize)
                .collect();
                path.var("weight", weight_size.as_slice(), ws_init)
            };

            Ok(ConvND {
                stride,
                padding,
                dilation,
                groups,
                weight: ws,
                bias: bs,
                transposed,
            })
        }
    }
}

mod conv_nd {
    use super::*;

    #[derive(Debug)]
    pub struct ConvND {
        pub(super) stride: Vec<i64>,
        pub(super) padding: Vec<i64>,
        pub(super) dilation: Vec<i64>,
        pub(super) groups: i64,
        pub(super) weight: Tensor,
        pub(super) bias: Option<Tensor>,
        pub(super) transposed: bool,
    }

    impl ConvND {
        pub fn set_trainable(&self, trainable: bool) {
            let Self { weight, bias, .. } = self;
            let _ = weight.set_requires_grad(trainable);
            if let Some(bias) = &bias {
                let _ = bias.set_requires_grad(trainable);
            }
        }

        pub fn stride(&self) -> &[i64] {
            &self.stride
        }

        pub fn forward(&self, input: &Tensor) -> Tensor {
            self.forward_ext(input, None)
        }

        pub fn forward_ext(&self, input: &Tensor, output_padding: Option<&[i64]>) -> Tensor {
            let Self {
                ref stride,
                ref padding,
                ref dilation,
                groups,
                ref weight,
                ref bias,
                transposed,
            } = *self;
            let ndims = stride.len();

            input.convolution(
                weight,
                bias.as_ref(),
                &stride,
                &padding,
                &dilation,
                transposed,
                output_padding.unwrap_or(&vec![0; ndims]),
                groups,
            )
        }
    }
}
