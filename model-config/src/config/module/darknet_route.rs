use super::*;
use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DarknetRoute {
    pub name: Option<ModuleName>,
    pub from: Option<ModulePath>,
    pub group_id: usize,
    pub num_groups: usize,
}

impl ModuleEx for DarknetRoute {
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
