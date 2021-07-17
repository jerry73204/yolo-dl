use crate::common::*;

pub use conv_init::*;
pub use conv_nd_::*;
pub use conv_nd_grad::*;

mod conv_init {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct ConvNDInit<S>
    where
        S: AsRef<[usize]>,
    {
        pub ksize: S,
        pub stride: S,
        pub padding: S,
        pub dilation: S,
        pub groups: usize,
        pub bias: bool,
        pub transposed: bool,
        pub ws_init: nn::Init,
        pub bs_init: nn::Init,
    }

    pub type Conv1DInit = ConvNDInit<[usize; 1]>;
    pub type Conv2DInit = ConvNDInit<[usize; 2]>;
    pub type Conv3DInit = ConvNDInit<[usize; 3]>;
    pub type Conv4DInit = ConvNDInit<[usize; 4]>;
    pub type ConvNDInitDyn = ConvNDInit<Vec<usize>>;

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

    impl<S> ConvNDInit<S>
    where
        S: AsRef<[usize]>,
    {
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
                ksize: ksize.as_ref().into(),
                stride: stride.as_ref().into(),
                padding: padding.as_ref().into(),
                dilation: dilation.as_ref().into(),
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
                ksize.as_ref().len() == stride.as_ref().len()
                    && ksize.as_ref().len() == padding.as_ref().len()
                    && ksize.as_ref().len() == dilation.as_ref().len(),
                "parameter dimension mismatch"
            );

            Ok(ksize.as_ref().len())
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
            let ksize: Vec<i64> = ksize.as_ref().iter().map(|&v| v as i64).collect();
            let stride: Vec<i64> = stride.as_ref().iter().map(|&v| v as i64).collect();
            let padding: Vec<i64> = padding.as_ref().iter().map(|&v| v as i64).collect();
            let dilation: Vec<i64> = dilation.as_ref().iter().map(|&v| v as i64).collect();
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

mod conv_nd_ {
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

        pub fn grad(&self) -> ConvNDGrad {
            let Self { weight, bias, .. } = self;

            ConvNDGrad {
                weight: weight.grad(),
                bias: bias.as_ref().map(Tensor::grad),
            }
        }
    }
}

mod conv_nd_grad {
    use super::*;

    #[derive(Debug, TensorLike)]
    pub struct ConvNDGrad {
        pub weight: Tensor,
        pub bias: Option<Tensor>,
    }
}
