use super::*;
use crate::{common::*, config::misc::Shape};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Input {
    pub name: ModuleName,
    pub shape: Shape,
}

impl ModuleEx for Input {
    fn name(&self) -> Option<&ModuleName> {
        Some(&self.name)
    }

    fn input_paths(&self) -> ModuleInput<'_> {
        ModuleInput::PlaceHolder
    }

    fn output_shape(&self, input_shape: ShapeInput<'_>) -> Option<ShapeOutput> {
        // expect a none shape
        let output_shape = match input_shape {
            ShapeInput::SingleTensor(input_shape) => input_shape.equalize(&self.shape)?,
            ShapeInput::PlaceHolder => self.shape.clone(),
            _ => return None,
        };

        Some(output_shape.into())
    }
}
