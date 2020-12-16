use super::module::{
    Bottleneck, BottleneckCsp, Concat, ConvBlock, Detect, Focus, Input, Output, Spp, UpSample,
};
use crate::{common::*, utils};

pub trait LayerEx {
    fn input_names(&self) -> Cow<'_, [&str]>;
}

#[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
#[derivative(Hash)]
pub struct Model(
    #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<String, Layer, _>"))]
    IndexMap<String, Layer>,
);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Layer {
    pub name: Option<String>,
    #[serde(flatten)]
    pub kind: LayerKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum LayerKind {
    Input(Input),
    Output(Output),
    Focus(Focus),
    ConvBlock(ConvBlock),
    Bottleneck(Bottleneck),
    BottleneckCsp(BottleneckCsp),
    Spp(Spp),
    UpSample(UpSample),
    Concat(Concat),
    Detect(Detect),
}

impl Layer {
    pub fn output_shape(&self, _input_shape: &[usize]) -> Option<Vec<usize>> {
        // match &self.kind {
        //     LayerKind::Input(layer) => layer.output_shape(input_shape),
        //     LayerKind::Focus(layer) => layer.output_shape(input_shape),
        //     LayerKind::ConvBlock(layer) => layer.output_shape(input_shape),
        //     LayerKind::Bottleneck(layer) => layer.output_shape(input_shape),
        //     LayerKind::BottleneckCsp(layer) => layer.output_shape(input_shape),
        //     LayerKind::Spp(layer) => layer.output_shape(input_shape),
        //     LayerKind::UpSample(layer) => layer.output_shape(input_shape),
        //     LayerKind::Concat(layer) => layer.output_shape(input_shape),
        //     LayerKind::Detect(layer) => layer.output_shape(input_shape),
        // }
        todo!();
    }
}

impl LayerEx for Layer {
    fn input_names(&self) -> Cow<'_, [&str]> {
        match &self.kind {
            LayerKind::Input(layer) => layer.input_names(),
            LayerKind::Output(layer) => layer.input_names(),
            LayerKind::Focus(layer) => layer.input_names(),
            LayerKind::ConvBlock(layer) => layer.input_names(),
            LayerKind::Bottleneck(layer) => layer.input_names(),
            LayerKind::BottleneckCsp(layer) => layer.input_names(),
            LayerKind::Spp(layer) => layer.input_names(),
            LayerKind::UpSample(layer) => layer.input_names(),
            LayerKind::Concat(layer) => layer.input_names(),
            LayerKind::Detect(layer) => layer.input_names(),
        }
    }
}
