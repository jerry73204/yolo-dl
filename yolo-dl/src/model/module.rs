use super::*;
use crate::common::*;

pub use bottleneck::*;
pub use bottleneck_csp::*;
pub use concat_2d::*;
pub use conv_block::*;
pub use conv_bn_2d::*;
pub use dark_csp_2d::*;
pub use detect::*;
pub use detect_2d::*;
pub use focus::*;
pub use input::*;
pub use merge_detect_2d::*;
pub use module::*;
pub use module_input::*;
pub use spp::*;
pub use spp_csp_2d::*;
pub use sum_2d::*;
pub use up_sample_2d::*;

mod module {
    pub use super::*;

    #[derive(Derivative)]
    #[derivative(Debug)]
    pub enum Module {
        Input(Input),
        ConvBn2D(ConvBn2D),
        UpSample2D(UpSample2D),
        Sum2D(Sum2D),
        Concat2D(Concat2D),
        DarkCsp2D(DarkCsp2D),
        SppCsp2D(SppCsp2D),
        Detect(DetectModule),
        Detect2D(Detect2D),
        MergeDetect2D(MergeDetect2D),
        FnSingle(
            #[derivative(Debug = "ignore")] Box<dyn 'static + Fn(&Tensor, bool) -> Tensor + Send>,
        ),
        FnIndexed(
            #[derivative(Debug = "ignore")]
            Box<dyn 'static + Fn(&[&Tensor], bool) -> Tensor + Send>,
        ),
    }

    impl Module {
        pub fn forward_t<'a, 'b>(
            &mut self,
            input: impl Into<ModuleInput<'a>>,
            train: bool,
            image_size: impl Into<Option<&'b PixelSize<i64>>>,
        ) -> Result<ModuleOutput> {
            let input = input.into();

            let output: ModuleOutput = match self {
                Self::Input(module) => module
                    .forward(input.tensor().ok_or_else(|| format_err!("TODO"))?)?
                    .into(),
                Self::Detect(module) => module
                    .forward_t(
                        &input.indexed_tensor().ok_or_else(|| format_err!("TODO"))?,
                        train,
                        image_size
                            .into()
                            .ok_or_else(|| format_err!("image size not provided"))?,
                    )
                    .into(),
                Self::ConvBn2D(module) => module
                    .forward_t(input.tensor().ok_or_else(|| format_err!("TODO"))?, train)
                    .into(),
                Self::UpSample2D(module) => module
                    .forward(input.tensor().ok_or_else(|| format_err!("TODO"))?)?
                    .into(),
                Self::Sum2D(module) => module
                    .forward(&input.indexed_tensor().ok_or_else(|| format_err!("TODO"))?)?
                    .into(),
                Self::Concat2D(module) => module
                    .forward(&input.indexed_tensor().ok_or_else(|| format_err!("TODO"))?)?
                    .into(),
                Self::DarkCsp2D(module) => module
                    .forward_t(input.tensor().ok_or_else(|| format_err!("TODO"))?, train)
                    .into(),
                Self::SppCsp2D(module) => module
                    .forward_t(input.tensor().ok_or_else(|| format_err!("TODO"))?, train)
                    .into(),
                Self::Detect2D(module) => module
                    .forward(input.tensor().ok_or_else(|| format_err!("TODO"))?)?
                    .into(),
                Self::MergeDetect2D(module) => module
                    .forward(
                        &input
                            .indexed_detect_2d()
                            .ok_or_else(|| format_err!("TODO"))?,
                    )?
                    .into(),
                Self::FnSingle(module) => {
                    module(input.tensor().ok_or_else(|| format_err!("TODO"))?, train).into()
                }
                Self::FnIndexed(module) => module(
                    &input.indexed_tensor().ok_or_else(|| format_err!("TODO"))?,
                    train,
                )
                .into(),
            };
            Ok(output)
        }
    }
}

mod module_input {
    pub use super::*;

    #[derive(Debug, Clone)]
    pub enum DataKind<'a> {
        Tensor(&'a Tensor),
        Detect2D(&'a Detect2DOutput),
    }

    impl<'a> DataKind<'a> {
        pub fn tensor(&self) -> Option<&Tensor> {
            match self {
                Self::Tensor(tensor) => Some(tensor),
                _ => None,
            }
        }

        pub fn detect_2d(&self) -> Option<&Detect2DOutput> {
            match self {
                Self::Detect2D(detect) => Some(detect),
                _ => None,
            }
        }
    }

    impl<'a> From<&'a Tensor> for DataKind<'a> {
        fn from(from: &'a Tensor) -> Self {
            Self::Tensor(from)
        }
    }

    impl<'a> From<&'a Detect2DOutput> for DataKind<'a> {
        fn from(from: &'a Detect2DOutput) -> Self {
            Self::Detect2D(from)
        }
    }

    #[derive(Debug, Clone)]
    pub enum ModuleInput<'a> {
        None,
        Single(DataKind<'a>),
        Indexed(Vec<DataKind<'a>>),
    }

    impl<'a> ModuleInput<'a> {
        pub fn tensor(&self) -> Option<&Tensor> {
            match self {
                Self::Single(DataKind::Tensor(tensor)) => Some(tensor),
                _ => None,
            }
        }

        pub fn detect_2d(&self) -> Option<&Detect2DOutput> {
            match self {
                Self::Single(DataKind::Detect2D(detect)) => Some(detect),
                _ => None,
            }
        }

        pub fn indexed_tensor(&self) -> Option<Vec<&Tensor>> {
            match self {
                Self::Indexed(indexed) => {
                    let tensors: Option<Vec<_>> =
                        indexed.iter().map(|data| data.tensor()).collect();
                    tensors
                }
                _ => None,
            }
        }

        pub fn indexed_detect_2d(&self) -> Option<Vec<&Detect2DOutput>> {
            match self {
                Self::Indexed(indexed) => {
                    let detects: Option<Vec<_>> =
                        indexed.iter().map(|data| data.detect_2d()).collect();
                    detects
                }
                _ => None,
            }
        }
    }

    impl<'a> From<&'a Tensor> for ModuleInput<'a> {
        fn from(from: &'a Tensor) -> Self {
            Self::Single(DataKind::from(from))
        }
    }

    impl<'a, 'b> From<&'b [&'a Tensor]> for ModuleInput<'a> {
        fn from(from: &'b [&'a Tensor]) -> Self {
            Self::Indexed(
                from.iter()
                    .cloned()
                    .map(|tensor| DataKind::from(tensor))
                    .collect(),
            )
        }
    }

    impl<'a> From<&'a Detect2DOutput> for ModuleInput<'a> {
        fn from(from: &'a Detect2DOutput) -> Self {
            Self::Single(DataKind::from(from))
        }
    }

    impl<'a, 'b> From<&'b [&'a Detect2DOutput]> for ModuleInput<'a> {
        fn from(from: &'b [&'a Detect2DOutput]) -> Self {
            Self::Indexed(
                from.iter()
                    .cloned()
                    .map(|tensor| DataKind::from(tensor))
                    .collect(),
            )
        }
    }

    impl<'a> TryFrom<&'a ModuleOutput> for ModuleInput<'a> {
        type Error = Error;

        fn try_from(from: &'a ModuleOutput) -> Result<Self, Self::Error> {
            let input: Self = match from {
                ModuleOutput::Tensor(tensor) => tensor.into(),
                ModuleOutput::Detect2D(detect) => detect.into(),
                _ => bail!("TODO"),
            };
            Ok(input)
        }
    }

    impl<'a, 'b> TryFrom<&'b [&'a ModuleOutput]> for ModuleInput<'a> {
        type Error = Error;

        fn try_from(from: &'b [&'a ModuleOutput]) -> Result<Self, Self::Error> {
            let first = from.first().ok_or_else(|| format_err!("TODO"))?;
            let input = match first {
                ModuleOutput::Tensor(_) => {
                    let input: Option<Vec<DataKind<'_>>> = from
                        .iter()
                        .cloned()
                        .map(|output| {
                            let tensor = output.as_tensor()?;
                            Some(tensor.into())
                        })
                        .collect();
                    Self::Indexed(input.ok_or_else(|| format_err!("TODO"))?)
                }
                ModuleOutput::Detect2D(_) => {
                    let input: Option<Vec<DataKind<'_>>> = from
                        .iter()
                        .cloned()
                        .map(|output| {
                            let tensor = output.as_detect_2d()?;
                            Some(tensor.into())
                        })
                        .collect();
                    Self::Indexed(input.ok_or_else(|| format_err!("TODO"))?)
                }
                _ => bail!("TODO"),
            };
            Ok(input)
        }
    }

    #[derive(Debug)]
    pub enum ModuleOutput {
        Tensor(Tensor),
        Yolo(YoloOutput),
        Detect2D(Detect2DOutput),
        MergeDetect2D(MergeDetect2DOutput),
    }

    impl ModuleOutput {
        pub fn as_tensor(&self) -> Option<&Tensor> {
            match self {
                Self::Tensor(tensor) => Some(tensor),
                _ => None,
            }
        }

        pub fn tensor(self) -> Option<Tensor> {
            match self {
                Self::Tensor(tensor) => Some(tensor),
                _ => None,
            }
        }

        pub fn yolo(self) -> Option<YoloOutput> {
            match self {
                Self::Yolo(detect) => Some(detect),
                _ => None,
            }
        }

        pub fn as_detect_2d(&self) -> Option<&Detect2DOutput> {
            match self {
                Self::Detect2D(detect) => Some(detect),
                _ => None,
            }
        }

        pub fn detect_2d(self) -> Option<Detect2DOutput> {
            match self {
                Self::Detect2D(detect) => Some(detect),
                _ => None,
            }
        }

        pub fn merge_detect_2d(self) -> Option<MergeDetect2DOutput> {
            match self {
                Self::MergeDetect2D(detect) => Some(detect),
                _ => None,
            }
        }
    }

    impl From<Tensor> for ModuleOutput {
        fn from(tensor: Tensor) -> Self {
            Self::Tensor(tensor)
        }
    }

    impl From<YoloOutput> for ModuleOutput {
        fn from(from: YoloOutput) -> Self {
            Self::Yolo(from)
        }
    }

    impl From<Detect2DOutput> for ModuleOutput {
        fn from(from: Detect2DOutput) -> Self {
            Self::Detect2D(from)
        }
    }

    impl From<MergeDetect2DOutput> for ModuleOutput {
        fn from(from: MergeDetect2DOutput) -> Self {
            Self::MergeDetect2D(from)
        }
    }
}

mod input {
    use super::*;

    #[derive(Debug)]
    pub struct Input {}

    impl Input {
        pub fn new() -> Self {
            Self {}
        }

        pub fn forward(&self, tensor: &Tensor) -> Result<Tensor> {
            // TODO: check shape
            Ok(tensor.shallow_clone())
        }
    }
}

mod up_sample_2d {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct UpSample2D {
        scale: f64,
    }

    impl UpSample2D {
        pub fn new(scale: f64) -> Result<Self> {
            ensure!(
                scale.is_finite() && scale.is_sign_positive(),
                "invalid scale value"
            );
            Ok(Self { scale })
        }

        pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
            let Self { scale } = *self;
            let (_b, _c, in_h, in_w) = input.size4()?;
            let out_h = (in_h as f64 * scale) as i64;
            let out_w = (in_w as f64 * scale) as i64;
            let output = input.upsample_nearest2d(&[out_h, out_w], None, None);
            Ok(output)
        }
    }
}

mod spp_csp_2d {
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
            let last_conv = ConvBn2DInit::new(mid_c, out_c, 1).build(path);
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
                        let p = k - 1;
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
                let xs = spp_conv_5.forward_t(&spp, train);
                xs
            };

            let merge = Tensor::cat(&[skip, spp], 1);
            let last = last_conv.forward_t(&merge, train);
            last
        }
    }
}

mod sum_2d {
    use super::*;

    #[derive(Debug)]
    pub struct Sum2D;

    impl Sum2D {
        pub fn forward<T>(&self, tensors: &[T]) -> Result<Tensor>
        where
            T: Borrow<Tensor>,
        {
            tensors.iter().try_for_each(|tensor| -> Result<_> {
                tensor.borrow().size4()?;
                Ok(())
            })?;
            ensure!(!tensors.is_empty(), "empty input is not allowed");
            let output = Tensor::cat(tensors, 1);
            Ok(output)
        }
    }
}

mod concat_2d {
    use super::*;

    #[derive(Debug)]
    pub struct Concat2D;

    impl Concat2D {
        pub fn forward<T>(&self, tensors: &[T]) -> Result<Tensor>
        where
            T: Borrow<Tensor>,
        {
            let mut iter = tensors.iter();
            let first = iter
                .next()
                .ok_or_else(|| format_err!("empty input is not allowed"))?
                .borrow()
                .shallow_clone();
            first.size4()?;

            let output = iter.try_fold(first, |acc, tensor| acc.f_add(tensor.borrow()))?;
            Ok(output)
        }
    }
}

mod conv_bn_2d {
    pub use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct ConvBn2DInit {
        pub in_c: usize,
        pub out_c: usize,
        pub k: usize,
        pub s: usize,
        pub p: usize,
        pub d: usize,
        pub g: usize,
        pub activation: Activation,
        pub batch_norm: bool,
    }

    impl ConvBn2DInit {
        pub fn new(in_c: usize, out_c: usize, k: usize) -> Self {
            Self {
                in_c,
                out_c,
                k,
                s: 1,
                p: k / 2,
                d: 1,
                g: 1,
                activation: Activation::Mish,
                batch_norm: true,
            }
        }

        pub fn build<'p, P>(self, path: P) -> ConvBn2D
        where
            P: Borrow<nn::Path<'p>>,
        {
            let path = path.borrow();

            let Self {
                in_c,
                out_c,
                k,
                s,
                p,
                d,
                g,
                activation,
                batch_norm,
            } = self;

            let conv = nn::conv2d(
                path,
                in_c as i64,
                out_c as i64,
                k as i64,
                nn::ConvConfig {
                    stride: s as i64,
                    padding: p as i64,
                    dilation: d as i64,
                    groups: g as i64,
                    bias: false,
                    ..Default::default()
                },
            );
            let bn = if batch_norm {
                Some(nn::batch_norm2d(path, out_c as i64, Default::default()))
            } else {
                None
            };

            ConvBn2D {
                conv,
                bn,
                activation,
            }
        }
    }

    #[derive(Debug)]
    pub struct ConvBn2D {
        conv: nn::Conv2D,
        bn: Option<nn::BatchNorm>,
        activation: Activation,
    }

    impl ConvBn2D {
        pub fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
            let Self {
                ref conv,
                ref bn,
                activation,
            } = *self;

            let xs = xs.apply(conv).activation(activation);

            let xs = match bn {
                Some(bn) => xs.apply_t(bn, train),
                None => xs,
            };

            xs
        }
    }
}

mod conv_block {
    pub use super::*;

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
                path,
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
            let bn = nn::batch_norm2d(path, out_c as i64, Default::default());

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
}

mod bottleneck {
    pub use super::*;

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
            .build(path);
            let conv2 = ConvBlockInit {
                k: 3,
                s: 1,
                g,
                ..ConvBlockInit::new(intermediate_channels, out_c)
            }
            .build(path);
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
}

mod bottleneck_csp {
    pub use super::*;

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
}

mod spp {
    pub use super::*;

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
            .build(path);

            let conv2 = ConvBlockInit {
                k: 1,
                s: 1,
                ..ConvBlockInit::new(intermediate_channels * (ks.len() + 1), out_c)
            }
            .build(path);

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
}

mod focus {
    pub use super::*;

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
            .build(path);

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
}

mod dark_csp_2d {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct DarkCsp2DInit {
        pub in_c: usize,
        pub out_c: usize,
        pub repeat: usize,
        pub shortcut: bool,
        pub c_mul: R64,
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
            } = self;

            let mid_c = (in_c as f64 * c_mul.raw()).floor() as usize;

            let skip_conv = ConvBn2DInit::new(in_c, mid_c, 1).build(path);
            let merge_conv = ConvBn2DInit::new(in_c, out_c, 1).build(path);
            let before_repeat_conv = ConvBn2DInit::new(in_c, mid_c, 1).build(path);
            let after_repeat_conv = ConvBn2DInit::new(mid_c, mid_c, 1).build(path);

            let repeat_convs: Vec<_> = (0..repeat)
                .map(|_| {
                    let first_conv = ConvBn2DInit::new(mid_c, mid_c, 1).build(path);
                    let second_conv = ConvBn2DInit::new(mid_c, mid_c, 3).build(path);
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
}

mod detect {
    pub use super::*;

    #[derive(Debug, Clone)]
    pub struct DetectInit {
        pub num_classes: usize,
        pub anchors_list: Vec<Vec<PixelSize<usize>>>,
    }

    impl DetectInit {
        pub fn build<'p, P>(self, path: P) -> DetectModule
        where
            P: Borrow<nn::Path<'p>>,
        {
            let path = path.borrow();
            let device = path.device();

            let Self {
                num_classes,
                anchors_list,
            } = self;

            let anchors_list: Vec<Vec<_>> = anchors_list
                .into_iter()
                .map(|list| {
                    list.into_iter()
                        .map(|PixelSize { height, width, .. }| {
                            PixelSize::new(height as i64, width as i64)
                        })
                        .collect()
                })
                .collect();

            DetectModule {
                num_classes: num_classes as i64,
                anchors_list,
                device,
                cache: HashMap::new(),
            }
        }
    }

    #[derive(Debug)]
    pub struct DetectModule {
        num_classes: i64,
        anchors_list: Vec<Vec<PixelSize<i64>>>,
        device: Device,
        cache: HashMap<PixelSize<i64>, DetectModuleCache>,
    }

    impl DetectModule {
        pub fn forward_t(
            &mut self,
            tensors: &[&Tensor],
            _train: bool,
            image_size: &PixelSize<i64>,
        ) -> YoloOutput {
            debug_assert_eq!(tensors.len(), self.anchors_list.len());
            let num_classes = self.num_classes;
            let device = self.device;
            let num_entries = num_classes + 5;
            let (batch_size, _channels, _height, _width) = tensors[0].size4().unwrap();

            // load cached data
            let DetectModuleCache {
                positions_grids,
                anchor_sizes_list,
                anchor_sizes_grids,
            } = &*self.cache(tensors, image_size);

            // compute outputs
            let (cy_vec, cx_vec, h_vec, w_vec, objectness_vec, classification_vec, layer_meta) =
                izip!(
                    tensors.iter(),
                    positions_grids,
                    anchor_sizes_list,
                    anchor_sizes_grids
                )
                .scan(
                    0,
                    |base_flat_index, (xs, positions_grid, anchor_sizes, anchor_sizes_grid)| {
                        let (bsize, channels, feature_height, feature_width) = xs.size4().unwrap();
                        let num_anchors = anchor_sizes.len() as i64;
                        debug_assert_eq!(bsize, batch_size);
                        debug_assert_eq!(channels, num_anchors * num_entries);
                        let feature_size = GridSize::new(feature_height, feature_width);

                        // gride size in pixels
                        let grid_height = image_size.height as f64 / feature_height as f64;
                        let grid_width = image_size.width as f64 / feature_width as f64;

                        // transform outputs
                        let (cy, cx, h, w, objectness, classification) = {
                            // convert shape to [batch_size, n_entries, n_anchors, height, width]
                            let outputs = xs.view([
                                batch_size,
                                num_entries,
                                num_anchors,
                                feature_height,
                                feature_width,
                            ]);

                            // positions in grid units
                            let (cy_map, cx_map, h_map, w_map) = {
                                let sigmoid = outputs.i((.., 0..4, .., .., ..)).sigmoid();

                                let position =
                                    sigmoid.i((.., 0..2, .., .., ..)) * 2.0 - 0.5 + positions_grid;
                                let cy_map = position.i((.., 0..1, .., .., ..));
                                let cx_map = position.i((.., 1..2, .., .., ..));

                                // bbox sizes in grid units
                                let size = sigmoid.i((.., 2..4, .., .., ..)) * anchor_sizes_grid;
                                let h_map = size.i((.., 0..1, .., .., ..));
                                let w_map = size.i((.., 1..2, .., .., ..));

                                (cy_map, cx_map, h_map, w_map)
                            };

                            // objectness
                            let objectness_map = outputs.i((.., 4..5, .., .., ..));

                            // sparse classification
                            let classification_map = outputs.i((.., 5.., .., .., ..));

                            let cy_flat = cy_map.view([batch_size, 1, -1]);
                            let cx_flat = cx_map.view([batch_size, 1, -1]);
                            let h_flat = h_map.view([batch_size, 1, -1]);
                            let w_flat = w_map.view([batch_size, 1, -1]);
                            let objectness_flat = objectness_map.view([batch_size, 1, -1]);
                            let classification_flat =
                                classification_map.view([batch_size, num_classes, -1]);

                            (
                                cy_flat,
                                cx_flat,
                                h_flat,
                                w_flat,
                                objectness_flat,
                                classification_flat,
                            )
                        };

                        // save feature anchors and shapes
                        let layer_meta = {
                            let begin_flat_index = *base_flat_index;
                            *base_flat_index += num_anchors * feature_height * feature_width;
                            let end_flat_index = *base_flat_index;

                            // compute base flat index
                            let layer_meta = LayerMeta {
                                feature_size: feature_size.to_owned(),
                                grid_size: PixelSize::new(
                                    R64::new(grid_height),
                                    R64::new(grid_width),
                                ),
                                anchors: anchor_sizes
                                    .iter()
                                    .map(|size| size.map(|&val| R64::new(val)))
                                    .collect(),
                                // begin_flat_index,
                                // end_flat_index,
                                flat_index_range: begin_flat_index..end_flat_index,
                            };
                            layer_meta
                        };

                        Some((cy, cx, h, w, objectness, classification, layer_meta))
                    },
                )
                .unzip_n_vec();

            let cy = Tensor::cat(&cy_vec, 2);
            let cx = Tensor::cat(&cx_vec, 2);
            let h = Tensor::cat(&h_vec, 2);
            let w = Tensor::cat(&w_vec, 2);
            let objectness = Tensor::cat(&objectness_vec, 2);
            let classification = Tensor::cat(&classification_vec, 2);

            YoloOutput {
                image_size: image_size.to_owned(),
                batch_size,
                num_classes,
                device,
                cy,
                cx,
                height: h,
                width: w,
                objectness,
                classification,
                layer_meta,
            }
        }

        fn cache(
            &mut self,
            tensors: &[&Tensor],
            image_size: &PixelSize<i64>,
        ) -> &DetectModuleCache {
            let device = self.device;
            let anchors_list = self.anchors_list.clone();

            self.cache.entry(image_size.to_owned()).or_insert_with(|| {
                tch::no_grad(|| {
                    let positions_grids = {
                        tensors
                            .iter()
                            .map(|xs| {
                                let (_bsize, _channels, feature_height, feature_width) =
                                    xs.size4().unwrap();
                                let grid = {
                                    let grids = Tensor::meshgrid(&[
                                        Tensor::arange(feature_height, (Kind::Float, device)),
                                        Tensor::arange(feature_width, (Kind::Float, device)),
                                    ]);
                                    // corresponds to (batch * entry * anchor * height * width)
                                    Tensor::stack(&[&grids[0], &grids[1]], 0).view([
                                        1,
                                        2,
                                        1,
                                        feature_height,
                                        feature_width,
                                    ])
                                };
                                grid.set_requires_grad(false)
                            })
                            .collect_vec()
                    };

                    let anchor_sizes_list: Vec<Vec<GridSize<f64>>> = {
                        let PixelSize {
                            height: image_h,
                            width: image_w,
                            ..
                        } = *image_size;

                        anchors_list
                            .iter()
                            .zip_eq(tensors.iter().cloned())
                            .map(|(anchors, xs)| {
                                let (_bsize, _channels, feature_h, feature_w) = xs.size4().unwrap();

                                // gride size in pixels
                                let grid_h = image_h as f64 / feature_h as f64;
                                let grid_w = image_w as f64 / feature_w as f64;

                                // convert anchor sizes into grid units
                                anchors
                                    .iter()
                                    .cloned()
                                    .map(|anchor_size| {
                                        let PixelSize {
                                            height: anchor_h,
                                            width: anchor_w,
                                            ..
                                        } = anchor_size;

                                        GridSize::new(
                                            anchor_h as f64 / grid_h,
                                            anchor_w as f64 / grid_w,
                                        )
                                    })
                                    .collect_vec()
                            })
                            .collect_vec()
                    };

                    let anchor_sizes_grids = {
                        anchor_sizes_list
                            .iter()
                            .map(|anchor_sizes| {
                                let num_anchors = anchor_sizes.len();
                                let (anchor_h_vec, anchor_w_vec) = anchor_sizes
                                    .iter()
                                    .cloned()
                                    .map(|anchor_size| {
                                        let GridSize {
                                            height: anchor_h,
                                            width: anchor_w,
                                            ..
                                        } = anchor_size;
                                        (anchor_h as f32, anchor_w as f32)
                                    })
                                    .unzip_n_vec();

                                let grid = Tensor::stack(
                                    &[
                                        Tensor::of_slice(&anchor_h_vec),
                                        Tensor::of_slice(&anchor_w_vec),
                                    ],
                                    0,
                                )
                                // corresponds to (batch * entry * anchor * height * width)
                                .view([1, 2, num_anchors as i64, 1, 1])
                                .set_requires_grad(false)
                                .to_device(device);

                                grid
                            })
                            .collect_vec()
                    };

                    DetectModuleCache {
                        positions_grids,
                        anchor_sizes_list,
                        anchor_sizes_grids,
                    }
                })
            })
        }
    }

    #[derive(Debug)]
    struct DetectModuleCache {
        positions_grids: Vec<Tensor>,
        anchor_sizes_list: Vec<Vec<GridSize<f64>>>,
        anchor_sizes_grids: Vec<Tensor>,
    }
}

mod detect_2d {
    pub use super::*;

    #[derive(Debug, Clone)]
    pub struct Detect2DInit {
        pub num_classes: usize,
        pub anchors: Vec<RatioSize>,
    }

    impl Detect2DInit {
        pub fn build<'p, P>(self, path: P) -> Detect2D
        where
            P: Borrow<nn::Path<'p>>,
        {
            let path = path.borrow();
            let device = path.device();

            let Self {
                num_classes,
                anchors,
            } = self;

            Detect2D {
                num_classes,
                anchors,
                device,
                cache: None,
            }
        }
    }

    #[derive(Debug)]
    pub struct Detect2D {
        num_classes: usize,
        anchors: Vec<RatioSize>,
        device: Device,
        cache: Option<(GridSize<i64>, Cache)>,
    }

    impl Detect2D {
        pub fn forward(&mut self, tensor: &Tensor) -> Result<Detect2DOutput> {
            let Self {
                num_classes,
                ref anchors,
                ..
            } = *self;
            let (batch_size, channels, feature_h, feature_w) = tensor.size4()?;
            let feature_size = GridSize::new(feature_h, feature_w);
            let anchors = anchors.to_owned();

            // load cached data
            let Cache {
                y_offsets,
                x_offsets,
                anchor_heights,
                anchor_widths,
            } = self.cache(tensor)?.shallow_clone();

            // compute outputs
            let num_anchors = anchors.len() as i64;
            let num_entries = num_classes as i64 + 5;
            debug_assert_eq!(channels, num_anchors * num_entries);

            // convert shape to [batch_size, n_entries, n_anchors, height, width]
            let outputs = tensor.view([batch_size, num_entries, num_anchors, feature_h, feature_w]);

            // positions in grid units

            let cy = (outputs.i((.., 0..1, .., .., ..)).sigmoid() * 2.0 - 0.5) / feature_h as f64
                + y_offsets.view([1, 1, 1, feature_h, 1]);
            let cx = (outputs.i((.., 1..2, .., .., ..)).sigmoid() * 2.0 - 0.5) / feature_w as f64
                + x_offsets.view([1, 1, 1, 1, feature_w]);

            // bbox sizes in grid units
            let h =
                outputs.i((.., 2..3, .., .., ..)) * anchor_heights.view([1, 1, num_anchors, 1, 1]);
            let w =
                outputs.i((.., 3..4, .., .., ..)) * anchor_widths.view([1, 1, num_anchors, 1, 1]);

            // objectness
            let obj = outputs.i((.., 4..5, .., .., ..));

            // sparse classification
            let class = outputs.i((.., 5.., .., .., ..));

            Ok(Detect2DOutput {
                batch_size,
                num_classes,
                feature_size,
                anchors: anchors.to_owned(),
                cy,
                cx,
                h,
                w,
                obj,
                class,
            })
        }

        fn cache(&mut self, tensor: &Tensor) -> Result<&Cache> {
            tch::no_grad(move || -> Result<_> {
                let Self {
                    device,
                    ref anchors,
                    ref mut cache,
                    ..
                } = *self;

                let (_b, _c, feature_h, feature_w) = tensor.size4()?;
                let expect_size = GridSize::new(feature_h, feature_w);

                let is_hit = cache
                    .as_ref()
                    .map(|(size, _cache)| size == &expect_size)
                    .unwrap_or(false);

                if !is_hit {
                    let y_offsets = (Tensor::arange(feature_h, (Kind::Float, device))
                        / feature_h as f64)
                        .set_requires_grad(false);
                    let x_offsets = (Tensor::arange(feature_w, (Kind::Float, device))
                        / feature_w as f64)
                        .set_requires_grad(false);

                    let (anchor_heights, anchor_widths) = {
                        let num_anchors = anchors.len();
                        let (anchor_h_vec, anchor_w_vec) = anchors
                            .iter()
                            .cloned()
                            .map(|anchor_size| {
                                let RatioSize {
                                    height: anchor_h,
                                    width: anchor_w,
                                    ..
                                } = anchor_size;
                                (anchor_h.to_f64() as f32, anchor_w.to_f64() as f32)
                            })
                            .unzip_n_vec();

                        let anchor_heights = Tensor::of_slice(&anchor_h_vec)
                            .set_requires_grad(false)
                            .to_device(device);
                        let anchor_widths = Tensor::of_slice(&anchor_w_vec)
                            .set_requires_grad(false)
                            .to_device(device);

                        (anchor_heights, anchor_widths)
                    };

                    let new_cache = Cache {
                        y_offsets,
                        x_offsets,
                        anchor_heights,
                        anchor_widths,
                    };

                    *cache = Some((expect_size, new_cache));
                }

                let cache = cache.as_ref().map(|(_size, cache)| cache).unwrap();
                Ok(cache)
            })
        }
    }

    #[derive(Debug)]
    struct Cache {
        y_offsets: Tensor,
        x_offsets: Tensor,
        anchor_heights: Tensor,
        anchor_widths: Tensor,
    }

    #[derive(Debug, TensorLike)]
    pub struct Detect2DOutput {
        pub batch_size: i64,
        #[tensor_like(clone)]
        pub feature_size: GridSize<i64>,
        pub num_classes: usize,
        #[tensor_like(clone)]
        pub anchors: Vec<RatioSize>,
        pub cy: Tensor,
        pub cx: Tensor,
        pub h: Tensor,
        pub w: Tensor,
        pub obj: Tensor,
        pub class: Tensor,
    }
}

mod merge_detect_2d {
    pub use super::*;

    #[derive(Debug)]
    pub struct MergeDetect2D {}

    impl MergeDetect2D {
        pub fn new() -> Self {
            Self {}
        }

        pub fn forward(&mut self, detections: &[&Detect2DOutput]) -> Result<MergeDetect2DOutput> {
            // ensure consistent sizes
            let (batch_size_set, num_classes_set): (HashSet<i64>, HashSet<usize>) = detections
                .iter()
                .cloned()
                .map(|detection| {
                    let Detect2DOutput {
                        batch_size,
                        num_classes,
                        ..
                    } = *detection;

                    (batch_size, num_classes)
                })
                .unzip_n();

            ensure!(batch_size_set.len() == 1, "TODO");
            ensure!(num_classes_set.len() == 1, "TODO");

            // merge detections
            let (cy_vec, cx_vec, h_vec, w_vec, obj_vec, class_vec, info) = detections
                .iter()
                .cloned()
                .scan(0, |base_flat_index, detection| {
                    let Detect2DOutput {
                        batch_size,
                        num_classes,
                        ref feature_size,
                        ref anchors,
                        ref cy,
                        ref cx,
                        ref h,
                        ref w,
                        ref obj,
                        ref class,
                        ..
                    } = *detection;

                    let num_anchors = anchors.len();
                    let GridSize {
                        height: feature_h,
                        width: feature_w,
                        ..
                    } = *feature_size;

                    // flatten tensors
                    let cy_flat = cy.view([batch_size, 1, -1]);
                    let cx_flat = cx.view([batch_size, 1, -1]);
                    let h_flat = h.view([batch_size, 1, -1]);
                    let w_flat = w.view([batch_size, 1, -1]);
                    let obj_flat = obj.view([batch_size, 1, -1]);
                    let class_flat = class.view([batch_size, num_classes as i64, -1]);

                    // save feature anchors and shapes
                    let info = {
                        let begin_flat_index = *base_flat_index;
                        *base_flat_index += num_anchors as i64 * feature_h * feature_w;
                        let end_flat_index = *base_flat_index;

                        // compute base flat index
                        let info = DetectionInfo {
                            feature_size: feature_size.to_owned(),
                            anchors: anchors.to_owned(),
                            flat_index_range: begin_flat_index..end_flat_index,
                        };
                        info
                    };

                    Some((cy_flat, cx_flat, h_flat, w_flat, obj_flat, class_flat, info))
                })
                .unzip_n_vec();

            let cy = Tensor::cat(&cy_vec, 2);
            let cx = Tensor::cat(&cx_vec, 2);
            let h = Tensor::cat(&h_vec, 2);
            let w = Tensor::cat(&w_vec, 2);
            let obj = Tensor::cat(&obj_vec, 2);
            let class = Tensor::cat(&class_vec, 2);

            Ok(MergeDetect2DOutput {
                cy,
                cx,
                h,
                w,
                obj,
                class,
                info,
            })
        }
    }

    #[derive(Debug, TensorLike)]
    pub struct MergeDetect2DOutput {
        pub cy: Tensor,
        pub cx: Tensor,
        pub h: Tensor,
        pub w: Tensor,
        pub obj: Tensor,
        pub class: Tensor,
        pub info: Vec<DetectionInfo>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, TensorLike)]
    pub struct DetectionInfo {
        /// feature map size in grid units
        #[tensor_like(clone)]
        pub feature_size: GridSize<i64>,
        /// Anchros (height, width) in grid units
        #[tensor_like(clone)]
        pub anchors: Vec<RatioSize>,
        #[tensor_like(clone)]
        pub flat_index_range: Range<i64>,
    }
}
