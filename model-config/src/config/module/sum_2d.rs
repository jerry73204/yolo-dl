use super::*;
use crate::{common::*, config::misc::Shape};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Sum2D {
    pub name: Option<ModuleName>,
    pub from: Vec<ModulePath>,
}

impl ModuleEx for Sum2D {
    fn name(&self) -> Option<&ModuleName> {
        self.name.as_ref()
    }

    fn input_paths(&self) -> ModuleInput<'_> {
        self.from.as_slice().into()
    }

    fn output_shape(&self, input_shape: ShapeInput<'_>) -> Option<ShapeOutput> {
        let input_shapes: &[&Shape] = input_shape.indexed_tensors()?;
        let mut iter = input_shapes.iter().cloned();
        let first = iter.next()?.to_owned();
        let output_shape = iter.try_fold(first, |acc, shape| acc.equalize(shape))?;
        Some(output_shape.into())
    }
}
