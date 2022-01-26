use crate::{
    common::*, concat_2d::Concat2D, conv_bn_2d::ConvBn2D, dark_csp_2d::DarkCsp2D,
    darknet_route::DarknetRoute, darknet_shortcut::DarknetShortcut, deconv_bn_2d::DeconvBn2D,
    detect_2d::Detect2D, dynamic_pad_nd::DynamicPad, input::Input, max_pool::MaxPool,
    merge_detect_2d::MergeDetect2D, spp_csp_2d::SppCsp2D, sum_2d::Sum2D, up_sample_2d::UpSample2D,
};

pub use module_::*;
mod module_ {

    use super::*;

    #[derive(AsRefStr, Derivative)]
    #[derivative(Debug)]
    pub enum Module {
        Input(Input),
        Conv2D(nn::Conv2D),
        ConvBn2D(ConvBn2D),
        DeconvBn2D(DeconvBn2D),
        UpSample2D(UpSample2D),
        Sum2D(Sum2D),
        Concat2D(Concat2D),
        DarkCsp2D(Box<DarkCsp2D>),
        SppCsp2D(Box<SppCsp2D>),
        Detect2D(Detect2D),
        DarknetRoute(DarknetRoute),
        DarknetShortcut(DarknetShortcut),
        MaxPool(MaxPool),
        MergeDetect2D(MergeDetect2D),
        DynamicPad2D(DynamicPad<2>),
        FnSingle(
            #[derivative(Debug = "ignore")] Box<dyn 'static + Fn(&Tensor, bool) -> Tensor + Send>,
        ),
        FnIndexed(
            #[derivative(Debug = "ignore")]
            Box<dyn 'static + Fn(&[&Tensor], bool) -> Tensor + Send>,
        ),
    }

    impl From<nn::Conv2D> for Module {
        fn from(v: nn::Conv2D) -> Self {
            Self::Conv2D(v)
        }
    }

    impl From<DynamicPad<2>> for Module {
        fn from(v: DynamicPad<2>) -> Self {
            Self::DynamicPad2D(v)
        }
    }

    impl Module {
        pub fn is_merge_detect_2d(&self) -> bool {
            matches!(self, Self::MergeDetect2D(_))
        }

        pub fn as_merge_detect_2d(&self) -> Option<&MergeDetect2D> {
            match self {
                Self::MergeDetect2D(module) => Some(module),
                _ => None,
            }
        }

        pub fn as_detect_2d(&self) -> Option<&Detect2D> {
            match self {
                Self::Detect2D(module) => Some(module),
                _ => None,
            }
        }
    }

    impl From<DarknetRoute> for Module {
        fn from(v: DarknetRoute) -> Self {
            Self::DarknetRoute(v)
        }
    }

    impl From<DarknetShortcut> for Module {
        fn from(v: DarknetShortcut) -> Self {
            Self::DarknetShortcut(v)
        }
    }

    impl From<MaxPool> for Module {
        fn from(v: MaxPool) -> Self {
            Self::MaxPool(v)
        }
    }

    impl From<Box<dyn 'static + Fn(&[&Tensor], bool) -> Tensor + Send>> for Module {
        fn from(v: Box<dyn 'static + Fn(&[&Tensor], bool) -> Tensor + Send>) -> Self {
            Self::FnIndexed(v)
        }
    }

    impl From<Box<dyn 'static + Fn(&Tensor, bool) -> Tensor + Send>> for Module {
        fn from(v: Box<dyn 'static + Fn(&Tensor, bool) -> Tensor + Send>) -> Self {
            Self::FnSingle(v)
        }
    }

    impl From<MergeDetect2D> for Module {
        fn from(v: MergeDetect2D) -> Self {
            Self::MergeDetect2D(v)
        }
    }

    impl From<Detect2D> for Module {
        fn from(v: Detect2D) -> Self {
            Self::Detect2D(v)
        }
    }

    impl From<SppCsp2D> for Module {
        fn from(v: SppCsp2D) -> Self {
            Self::SppCsp2D(Box::new(v))
        }
    }

    impl From<DarkCsp2D> for Module {
        fn from(v: DarkCsp2D) -> Self {
            Self::DarkCsp2D(Box::new(v))
        }
    }

    impl From<Concat2D> for Module {
        fn from(v: Concat2D) -> Self {
            Self::Concat2D(v)
        }
    }

    impl From<Sum2D> for Module {
        fn from(v: Sum2D) -> Self {
            Self::Sum2D(v)
        }
    }

    impl From<UpSample2D> for Module {
        fn from(v: UpSample2D) -> Self {
            Self::UpSample2D(v)
        }
    }

    impl From<DeconvBn2D> for Module {
        fn from(v: DeconvBn2D) -> Self {
            Self::DeconvBn2D(v)
        }
    }

    impl From<ConvBn2D> for Module {
        fn from(v: ConvBn2D) -> Self {
            Self::ConvBn2D(v)
        }
    }

    impl From<Input> for Module {
        fn from(v: Input) -> Self {
            Self::Input(v)
        }
    }

    impl Module {
        pub fn forward_t<'a>(
            &mut self,
            input: impl Into<ModuleInput<'a>>,
            train: bool,
        ) -> Result<ModuleOutput> {
            let input = input.into();

            let output: ModuleOutput = match self {
                Self::Input(module) => module
                    .forward(input.tensor().ok_or_else(|| format_err!("TODO"))?)?
                    .into(),
                Self::ConvBn2D(module) => module
                    .forward_t(input.tensor().ok_or_else(|| format_err!("TODO"))?, train)
                    .into(),
                Self::DeconvBn2D(module) => module
                    .forward_t(input.tensor().ok_or_else(|| format_err!("TODO"))?, train)
                    .into(),
                Self::UpSample2D(module) => module
                    .forward(input.tensor().ok_or_else(|| format_err!("TODO"))?)?
                    .into(),
                Self::Sum2D(module) => module
                    .forward(input.indexed_tensor().ok_or_else(|| format_err!("TODO"))?)?
                    .into(),
                Self::Concat2D(module) => module
                    .forward(input.indexed_tensor().ok_or_else(|| format_err!("TODO"))?)?
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
                        input
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
                Module::DynamicPad2D(module) => module
                    .forward(input.tensor().ok_or_else(|| format_err!("TODO"))?)
                    .into(),
                Module::Conv2D(module) => module
                    .forward(input.tensor().ok_or_else(|| format_err!("TODO"))?)
                    .into(),
                Module::DarknetRoute(_) => {
                    todo!();
                }
                Module::DarknetShortcut(_) => {
                    todo!();
                }
                Module::MaxPool(_) => {
                    todo!();
                }
            };

            Ok(output)
        }

        pub fn clamp_running_var(&mut self) {
            match self {
                Self::ConvBn2D(module) => module.clamp_running_var(),
                Self::DeconvBn2D(module) => module.clamp_running_var(),
                Self::DarkCsp2D(module) => module.clamp_running_var(),
                Self::SppCsp2D(module) => module.clamp_running_var(),
                Self::Input(_)
                | Self::UpSample2D(_)
                | Self::Sum2D(_)
                | Self::Concat2D(_)
                | Self::Detect2D(_)
                | Self::MergeDetect2D(_)
                | Self::FnSingle(_)
                | Self::FnIndexed(_)
                | Module::DarknetRoute(_)
                | Module::DarknetShortcut(_)
                | Module::MaxPool(_)
                | Module::DynamicPad2D(_)
                | Module::Conv2D(_) => {}
            }
        }

        pub fn denormalize(&mut self) {
            match self {
                Self::ConvBn2D(module) => module.denormalize(),
                Self::DeconvBn2D(module) => module.denormalize(),
                Self::DarkCsp2D(module) => module.denormalize(),
                Self::SppCsp2D(module) => module.denormalize(),
                Self::Input(_)
                | Self::UpSample2D(_)
                | Self::Sum2D(_)
                | Self::Concat2D(_)
                | Self::Detect2D(_)
                | Self::MergeDetect2D(_)
                | Self::FnSingle(_)
                | Self::FnIndexed(_)
                | Module::DarknetRoute(_)
                | Module::DarknetShortcut(_)
                | Module::MaxPool(_)
                | Module::DynamicPad2D(_)
                | Module::Conv2D(_) => {}
            }
        }
    }
}

pub use module_input::*;
mod module_input {
    use super::*;

    #[derive(Debug, Clone)]
    pub enum DataKind<'a> {
        Tensor(&'a Tensor),
        Detect2D(&'a tch_goodies::DenseDetectionTensor),
    }

    impl<'a> DataKind<'a> {
        pub fn tensor(&self) -> Option<&Tensor> {
            match self {
                Self::Tensor(tensor) => Some(tensor),
                _ => None,
            }
        }

        pub fn detect_2d(&self) -> Option<&tch_goodies::DenseDetectionTensor> {
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

    impl<'a> From<&'a tch_goodies::DenseDetectionTensor> for DataKind<'a> {
        fn from(from: &'a tch_goodies::DenseDetectionTensor) -> Self {
            Self::Detect2D(from)
        }
    }

    impl<'a> TryFrom<&'a ModuleOutput> for DataKind<'a> {
        type Error = Error;

        fn try_from(from: &'a ModuleOutput) -> Result<Self, Self::Error> {
            let kind = match from {
                ModuleOutput::Tensor(tensor) => Self::Tensor(tensor),
                ModuleOutput::Detect2D(detect) => Self::Detect2D(detect),
                _ => bail!("TODO"),
            };
            Ok(kind)
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

        pub fn detect_2d(&self) -> Option<&tch_goodies::DenseDetectionTensor> {
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

        pub fn indexed_detect_2d(&self) -> Option<Vec<&tch_goodies::DenseDetectionTensor>> {
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
            Self::Indexed(from.iter().cloned().map(DataKind::from).collect())
        }
    }

    impl<'a> From<&'a [Tensor]> for ModuleInput<'a> {
        fn from(from: &'a [Tensor]) -> Self {
            Self::Indexed(from.iter().map(DataKind::from).collect())
        }
    }

    impl<'a> From<&'a tch_goodies::DenseDetectionTensor> for ModuleInput<'a> {
        fn from(from: &'a tch_goodies::DenseDetectionTensor) -> Self {
            Self::Single(DataKind::from(from))
        }
    }

    impl<'a, 'b> From<&'b [&'a tch_goodies::DenseDetectionTensor]> for ModuleInput<'a> {
        fn from(from: &'b [&'a tch_goodies::DenseDetectionTensor]) -> Self {
            Self::Indexed(from.iter().cloned().map(DataKind::from).collect())
        }
    }

    impl<'a> From<&'a [tch_goodies::DenseDetectionTensor]> for ModuleInput<'a> {
        fn from(from: &'a [tch_goodies::DenseDetectionTensor]) -> Self {
            Self::Indexed(from.iter().map(DataKind::from).collect())
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
            let kinds: Vec<DataKind> =
                from.iter().cloned().map(DataKind::try_from).try_collect()?;
            Ok(Self::Indexed(kinds))
        }
    }

    impl<'a> TryFrom<&'a [ModuleOutput]> for ModuleInput<'a> {
        type Error = Error;

        fn try_from(from: &'a [ModuleOutput]) -> Result<Self, Self::Error> {
            let kinds: Vec<DataKind> = from.iter().map(DataKind::try_from).try_collect()?;
            Ok(Self::Indexed(kinds))
        }
    }
}

pub use module_output::*;
mod module_output {
    use super::*;

    #[derive(Debug, TensorLike)]
    pub enum ModuleOutput {
        Tensor(Tensor),
        Detect2D(tch_goodies::DenseDetectionTensor),
        MergeDetect2D(tch_goodies::DenseDetectionTensorList),
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

        pub fn as_detect_2d(&self) -> Option<&tch_goodies::DenseDetectionTensor> {
            match self {
                Self::Detect2D(detect) => Some(detect),
                _ => None,
            }
        }

        pub fn detect_2d(self) -> Option<tch_goodies::DenseDetectionTensor> {
            match self {
                Self::Detect2D(detect) => Some(detect),
                _ => None,
            }
        }

        pub fn merge_detect_2d(self) -> Option<tch_goodies::DenseDetectionTensorList> {
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

    impl From<tch_goodies::DenseDetectionTensor> for ModuleOutput {
        fn from(from: tch_goodies::DenseDetectionTensor) -> Self {
            Self::Detect2D(from)
        }
    }

    impl From<tch_goodies::DenseDetectionTensorList> for ModuleOutput {
        fn from(from: tch_goodies::DenseDetectionTensorList) -> Self {
            Self::MergeDetect2D(from)
        }
    }
}
