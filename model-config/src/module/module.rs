use super::*;
use crate::common::*;

pub trait ModuleEx {
    fn name(&self) -> Option<&ModuleName>;
    fn input_paths(&self) -> ModuleInput<'_>;
    fn output_shape(&self, input_shape: ShapeInput<'_>) -> Option<ShapeOutput>;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, AsRefStr, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Module {
    Input(Input),
    ConvBn2D(ConvBn2D),
    DeconvBn2D(DeconvBn2D),
    DarkCsp2D(DarkCsp2D),
    SppCsp2D(SppCsp2D),
    UpSample2D(UpSample2D),
    Concat2D(Concat2D),
    Sum2D(Sum2D),
    Detect2D(Detect2D),
    GroupRef(GroupRef),
    MergeDetect2D(MergeDetect2D),
    DarknetRoute(DarknetRoute),
    DarknetShortcut(DarknetShortcut),
    MaxPool(MaxPool),
    Linear(Linear),
    DynamicPad2d(DynamicPad2D),
}

impl From<Linear> for Module {
    fn from(v: Linear) -> Self {
        Module::Linear(v)
    }
}

impl From<MaxPool> for Module {
    fn from(v: MaxPool) -> Self {
        Module::MaxPool(v)
    }
}

impl From<DarknetShortcut> for Module {
    fn from(v: DarknetShortcut) -> Self {
        Module::DarknetShortcut(v)
    }
}

impl From<DarknetRoute> for Module {
    fn from(v: DarknetRoute) -> Self {
        Module::DarknetRoute(v)
    }
}

impl From<MergeDetect2D> for Module {
    fn from(v: MergeDetect2D) -> Self {
        Module::MergeDetect2D(v)
    }
}

impl From<GroupRef> for Module {
    fn from(v: GroupRef) -> Self {
        Module::GroupRef(v)
    }
}

impl From<Detect2D> for Module {
    fn from(v: Detect2D) -> Self {
        Module::Detect2D(v)
    }
}

impl From<Sum2D> for Module {
    fn from(v: Sum2D) -> Self {
        Module::Sum2D(v)
    }
}

impl From<Concat2D> for Module {
    fn from(v: Concat2D) -> Self {
        Module::Concat2D(v)
    }
}

impl From<UpSample2D> for Module {
    fn from(v: UpSample2D) -> Self {
        Module::UpSample2D(v)
    }
}

impl From<SppCsp2D> for Module {
    fn from(v: SppCsp2D) -> Self {
        Module::SppCsp2D(v)
    }
}

impl From<DarkCsp2D> for Module {
    fn from(v: DarkCsp2D) -> Self {
        Module::DarkCsp2D(v)
    }
}

impl From<DeconvBn2D> for Module {
    fn from(v: DeconvBn2D) -> Self {
        Module::DeconvBn2D(v)
    }
}

impl From<ConvBn2D> for Module {
    fn from(v: ConvBn2D) -> Self {
        Module::ConvBn2D(v)
    }
}

impl From<Input> for Module {
    fn from(v: Input) -> Self {
        Module::Input(v)
    }
}

impl ModuleEx for Module {
    fn name(&self) -> Option<&ModuleName> {
        match self {
            Module::Input(layer) => layer.name(),
            Module::ConvBn2D(layer) => layer.name(),
            Module::DeconvBn2D(layer) => layer.name(),
            Module::UpSample2D(layer) => layer.name(),
            Module::DarkCsp2D(layer) => layer.name(),
            Module::SppCsp2D(layer) => layer.name(),
            Module::Concat2D(layer) => layer.name(),
            Module::Sum2D(layer) => layer.name(),
            Module::Detect2D(layer) => layer.name(),
            Module::GroupRef(layer) => layer.name(),
            Module::MergeDetect2D(layer) => layer.name(),
            Module::DarknetRoute(layer) => layer.name(),
            Module::DarknetShortcut(layer) => layer.name(),
            Module::MaxPool(layer) => layer.name(),
            Module::Linear(layer) => layer.name(),
            Module::DynamicPad2d(layer) => layer.name(),
        }
    }

    fn input_paths(&self) -> ModuleInput<'_> {
        match self {
            Module::Input(layer) => layer.input_paths(),
            Module::ConvBn2D(layer) => layer.input_paths(),
            Module::DeconvBn2D(layer) => layer.input_paths(),
            Module::UpSample2D(layer) => layer.input_paths(),
            Module::DarkCsp2D(layer) => layer.input_paths(),
            Module::SppCsp2D(layer) => layer.input_paths(),
            Module::Concat2D(layer) => layer.input_paths(),
            Module::Sum2D(layer) => layer.input_paths(),
            Module::Detect2D(layer) => layer.input_paths(),
            Module::GroupRef(layer) => layer.input_paths(),
            Module::MergeDetect2D(layer) => layer.input_paths(),
            Module::DarknetRoute(layer) => layer.input_paths(),
            Module::DarknetShortcut(layer) => layer.input_paths(),
            Module::MaxPool(layer) => layer.input_paths(),
            Module::Linear(layer) => layer.input_paths(),
            Module::DynamicPad2d(layer) => layer.input_paths(),
        }
    }

    fn output_shape(&self, input_shape: ShapeInput<'_>) -> Option<ShapeOutput> {
        match self {
            Module::Input(layer) => layer.output_shape(input_shape),
            Module::ConvBn2D(layer) => layer.output_shape(input_shape),
            Module::DeconvBn2D(layer) => layer.output_shape(input_shape),
            Module::UpSample2D(layer) => layer.output_shape(input_shape),
            Module::DarkCsp2D(layer) => layer.output_shape(input_shape),
            Module::SppCsp2D(layer) => layer.output_shape(input_shape),
            Module::Concat2D(layer) => layer.output_shape(input_shape),
            Module::Sum2D(layer) => layer.output_shape(input_shape),
            Module::Detect2D(layer) => layer.output_shape(input_shape),
            Module::GroupRef(layer) => layer.output_shape(input_shape),
            Module::MergeDetect2D(layer) => layer.output_shape(input_shape),
            Module::DarknetRoute(layer) => layer.output_shape(input_shape),
            Module::DarknetShortcut(layer) => layer.output_shape(input_shape),
            Module::MaxPool(layer) => layer.output_shape(input_shape),
            Module::Linear(layer) => layer.output_shape(input_shape),
            Module::DynamicPad2d(layer) => layer.output_shape(input_shape),
        }
    }
}
