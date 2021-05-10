use super::*;
use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MaxPool {
    pub name: Option<ModuleName>,
    pub from: Option<ModulePath>,
    pub stride_x: usize,
    pub stride_y: usize,
    pub size: usize,
    pub padding: usize,
    pub maxpool_depth: bool,
}

impl ModuleEx for MaxPool {
    fn name(&self) -> Option<&ModuleName> {
        self.name.as_ref()
    }

    fn input_paths(&self) -> ModuleInput<'_> {
        self.from.as_ref().into()
    }

    fn output_shape(&self, _input_shape: ShapeInput<'_>) -> Option<ShapeOutput> {
        todo!();
    }
}
