use super::input::Input;
use crate::common::*;
use tch_goodies::module as goodies_mod;

pub use module::*;
pub use module_input::*;

mod module {
    use super::*;

    #[derive(AsRefStr, Derivative)]
    #[derivative(Debug)]
    pub enum Module {
        Input(Input),
        ConvBn2D(goodies_mod::ConvBn2D),
        DeconvBn2D(goodies_mod::DeconvBn2D),
        UpSample2D(goodies_mod::UpSample2D),
        Sum2D(goodies_mod::Sum2D),
        Concat2D(goodies_mod::Concat2D),
        DarkCsp2D(goodies_mod::DarkCsp2D),
        SppCsp2D(goodies_mod::SppCsp2D),
        Detect2D(goodies_mod::Detect2D),
        DarknetRoute(goodies_mod::DarknetRoute),
        DarknetShortcut(goodies_mod::DarknetShortcut),
        MaxPool(goodies_mod::MaxPool),
        MergeDetect2D(goodies_mod::MergeDetect2D),
        FnSingle(
            #[derivative(Debug = "ignore")] Box<dyn 'static + Fn(&Tensor, bool) -> Tensor + Send>,
        ),
        FnIndexed(
            #[derivative(Debug = "ignore")]
            Box<dyn 'static + Fn(&[&Tensor], bool) -> Tensor + Send>,
        ),
    }

    impl From<goodies_mod::DarknetRoute> for Module {
        fn from(v: goodies_mod::DarknetRoute) -> Self {
            Self::DarknetRoute(v)
        }
    }

    impl From<goodies_mod::DarknetShortcut> for Module {
        fn from(v: goodies_mod::DarknetShortcut) -> Self {
            Self::DarknetShortcut(v)
        }
    }

    impl From<goodies_mod::MaxPool> for Module {
        fn from(v: goodies_mod::MaxPool) -> Self {
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

    impl From<goodies_mod::MergeDetect2D> for Module {
        fn from(v: goodies_mod::MergeDetect2D) -> Self {
            Self::MergeDetect2D(v)
        }
    }

    impl From<goodies_mod::Detect2D> for Module {
        fn from(v: goodies_mod::Detect2D) -> Self {
            Self::Detect2D(v)
        }
    }

    impl From<goodies_mod::SppCsp2D> for Module {
        fn from(v: goodies_mod::SppCsp2D) -> Self {
            Self::SppCsp2D(v)
        }
    }

    impl From<goodies_mod::DarkCsp2D> for Module {
        fn from(v: goodies_mod::DarkCsp2D) -> Self {
            Self::DarkCsp2D(v)
        }
    }

    impl From<goodies_mod::Concat2D> for Module {
        fn from(v: goodies_mod::Concat2D) -> Self {
            Self::Concat2D(v)
        }
    }

    impl From<goodies_mod::Sum2D> for Module {
        fn from(v: goodies_mod::Sum2D) -> Self {
            Self::Sum2D(v)
        }
    }

    impl From<goodies_mod::UpSample2D> for Module {
        fn from(v: goodies_mod::UpSample2D) -> Self {
            Self::UpSample2D(v)
        }
    }

    impl From<goodies_mod::DeconvBn2D> for Module {
        fn from(v: goodies_mod::DeconvBn2D) -> Self {
            Self::DeconvBn2D(v)
        }
    }

    impl From<goodies_mod::ConvBn2D> for Module {
        fn from(v: goodies_mod::ConvBn2D) -> Self {
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

        pub fn clamp_bn_var(&mut self) {
            match self {
                Self::ConvBn2D(module) => module.clamp_bn_var(),
                Self::DeconvBn2D(module) => module.clamp_bn_var(),
                Self::DarkCsp2D(module) => module.clamp_bn_var(),
                Self::SppCsp2D(module) => module.clamp_bn_var(),
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
                | Module::MaxPool(_) => {}
            }
        }

        pub fn denormalize_bn(&mut self) {
            match self {
                Self::ConvBn2D(module) => module.denormalize_bn(),
                Self::DeconvBn2D(module) => module.denormalize_bn(),
                Self::DarkCsp2D(module) => module.denormalize_bn(),
                Self::SppCsp2D(module) => module.denormalize_bn(),
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
                | Module::MaxPool(_) => {}
            }
        }
    }
}

mod module_input {
    use super::*;

    #[derive(Debug, Clone)]
    pub enum DataKind<'a> {
        Tensor(&'a Tensor),
        Detect2D(&'a goodies_mod::Detect2DOutput),
    }

    impl<'a> DataKind<'a> {
        pub fn tensor(&self) -> Option<&Tensor> {
            match self {
                Self::Tensor(tensor) => Some(tensor),
                _ => None,
            }
        }

        pub fn detect_2d(&self) -> Option<&goodies_mod::Detect2DOutput> {
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

    impl<'a> From<&'a goodies_mod::Detect2DOutput> for DataKind<'a> {
        fn from(from: &'a goodies_mod::Detect2DOutput) -> Self {
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

        pub fn detect_2d(&self) -> Option<&goodies_mod::Detect2DOutput> {
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

        pub fn indexed_detect_2d(&self) -> Option<Vec<&goodies_mod::Detect2DOutput>> {
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

    impl<'a> From<&'a [Tensor]> for ModuleInput<'a> {
        fn from(from: &'a [Tensor]) -> Self {
            Self::Indexed(from.iter().map(|tensor| DataKind::from(tensor)).collect())
        }
    }

    impl<'a> From<&'a goodies_mod::Detect2DOutput> for ModuleInput<'a> {
        fn from(from: &'a goodies_mod::Detect2DOutput) -> Self {
            Self::Single(DataKind::from(from))
        }
    }

    impl<'a, 'b> From<&'b [&'a goodies_mod::Detect2DOutput]> for ModuleInput<'a> {
        fn from(from: &'b [&'a goodies_mod::Detect2DOutput]) -> Self {
            Self::Indexed(
                from.iter()
                    .cloned()
                    .map(|tensor| DataKind::from(tensor))
                    .collect(),
            )
        }
    }

    impl<'a> From<&'a [goodies_mod::Detect2DOutput]> for ModuleInput<'a> {
        fn from(from: &'a [goodies_mod::Detect2DOutput]) -> Self {
            Self::Indexed(from.iter().map(|output| DataKind::from(output)).collect())
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
            let kinds: Vec<DataKind> = from
                .iter()
                .cloned()
                .map(|output| DataKind::try_from(output))
                .try_collect()?;
            Ok(Self::Indexed(kinds))
        }
    }

    impl<'a> TryFrom<&'a [ModuleOutput]> for ModuleInput<'a> {
        type Error = Error;

        fn try_from(from: &'a [ModuleOutput]) -> Result<Self, Self::Error> {
            let kinds: Vec<DataKind> = from
                .iter()
                .map(|output| DataKind::try_from(output))
                .try_collect()?;
            Ok(Self::Indexed(kinds))
        }
    }

    #[derive(Debug, TensorLike)]
    pub enum ModuleOutput {
        Tensor(Tensor),
        Detect2D(goodies_mod::Detect2DOutput),
        MergeDetect2D(goodies_mod::MergeDetect2DOutput),
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

        pub fn as_detect_2d(&self) -> Option<&goodies_mod::Detect2DOutput> {
            match self {
                Self::Detect2D(detect) => Some(detect),
                _ => None,
            }
        }

        pub fn detect_2d(self) -> Option<goodies_mod::Detect2DOutput> {
            match self {
                Self::Detect2D(detect) => Some(detect),
                _ => None,
            }
        }

        pub fn merge_detect_2d(self) -> Option<goodies_mod::MergeDetect2DOutput> {
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

    impl From<goodies_mod::Detect2DOutput> for ModuleOutput {
        fn from(from: goodies_mod::Detect2DOutput) -> Self {
            Self::Detect2D(from)
        }
    }

    impl From<goodies_mod::MergeDetect2DOutput> for ModuleOutput {
        fn from(from: goodies_mod::MergeDetect2DOutput) -> Self {
            Self::MergeDetect2D(from)
        }
    }
}
