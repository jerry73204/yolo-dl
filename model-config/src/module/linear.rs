use super::*;
use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Linear {
    pub name: Option<ModuleName>,
    pub from: Option<ModulePath>,
    pub out: usize,
    pub bn: BatchNorm,
}

impl ModuleEx for Linear {
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
