use super::*;
use crate::{common::*, utils};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Derivative)]
#[derivative(Hash)]
pub enum ModuleInput<'a> {
    None,
    PlaceHolder,
    Infer,
    Path(&'a ModulePath),
    Indexed(&'a [ModulePath]),
    Named(
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<ModuleName, ModulePath, _>"))]
        &'a IndexMap<ModuleName, ModulePath>,
    ),
}

impl<'a> From<&'a ModulePath> for ModuleInput<'a> {
    fn from(from: &'a ModulePath) -> Self {
        Self::Path(from)
    }
}

impl<'a> From<&'a Option<ModulePath>> for ModuleInput<'a> {
    fn from(from: &'a Option<ModulePath>) -> Self {
        from.as_ref().into()
    }
}

impl<'a> From<Option<&'a ModulePath>> for ModuleInput<'a> {
    fn from(from: Option<&'a ModulePath>) -> Self {
        match from {
            Some(path) => path.into(),
            None => ModuleInput::Infer,
        }
    }
}

impl<'a> From<&'a [ModulePath]> for ModuleInput<'a> {
    fn from(from: &'a [ModulePath]) -> Self {
        Self::Indexed(from)
    }
}

impl<'a> From<&'a IndexMap<ModuleName, ModulePath>> for ModuleInput<'a> {
    fn from(from: &'a IndexMap<ModuleName, ModulePath>) -> Self {
        Self::Named(from)
    }
}
