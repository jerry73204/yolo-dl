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
        }
    }
}
