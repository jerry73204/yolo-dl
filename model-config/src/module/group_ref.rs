use super::*;
use crate::{common::*, group::GroupName, utils};

#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(PartialEq, Eq, Hash)]
pub struct GroupRef {
    pub name: ModuleName,
    #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<ModuleName, ModulePath, _>"))]
    pub from: IndexMap<ModuleName, ModulePath>,
    pub group: GroupName,
}

impl ModuleEx for GroupRef {
    fn name(&self) -> Option<&ModuleName> {
        Some(&self.name)
    }

    fn input_paths(&self) -> ModuleInput<'_> {
        (&self.from).into()
    }

    fn output_shape(&self, _input_shape: ShapeInput<'_>) -> Option<ShapeOutput> {
        None
    }
}
