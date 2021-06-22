use super::*;
use crate::common::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PaddingKind {
    Zero,
    Replication,
    Reflection,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DynamicPad2D {
    pub name: Option<ModuleName>,
    pub from: Option<ModulePath>,
    pub r#type: PaddingKind,
    pub l: usize,
    pub r: usize,
    pub t: usize,
    pub b: usize,
}

impl ModuleEx for DynamicPad2D {
    fn name(&self) -> Option<&ModuleName> {
        self.name.as_ref()
    }

    fn input_paths(&self) -> ModuleInput<'_> {
        self.from.as_ref().into()
    }

    fn output_shape(&self, input_shape: ShapeInput<'_>) -> Option<ShapeOutput> {
        let [b, c, h, w] = input_shape.single_tensor()?.size4()?;
        Some([b, c, h + self.t + self.b, w + self.l + self.r].into())
    }
}
