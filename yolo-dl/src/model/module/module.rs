use super::*;
use crate::common::*;

pub use module::*;
pub use module_input::*;

mod module {
    use super::*;

    #[derive(AsRefStr, Derivative)]
    #[derivative(Debug)]
    pub enum Module {
        Input(Input),
        ConvBn2D(ConvBn2D),
        DeconvBn2D(DeconvBn2D),
        UpSample2D(UpSample2D),
        Sum2D(Sum2D),
        Concat2D(Concat2D),
        DarkCsp2D(DarkCsp2D),
        SppCsp2D(SppCsp2D),
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
            };

            Ok(output)
        }
    }
}

mod module_input {
    use super::*;

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

    impl<'a> From<&'a [Tensor]> for ModuleInput<'a> {
        fn from(from: &'a [Tensor]) -> Self {
            Self::Indexed(from.iter().map(|tensor| DataKind::from(tensor)).collect())
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

    impl<'a> From<&'a [Detect2DOutput]> for ModuleInput<'a> {
        fn from(from: &'a [Detect2DOutput]) -> Self {
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
