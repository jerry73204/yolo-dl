use super::*;
use crate::{
    common::*,
    config::misc::{Dim, Size},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Detect2D {
    pub name: Option<ModuleName>,
    pub from: Option<ModulePath>,
    pub classes: usize,
    pub anchors: Vec<Size>,
}

impl ModuleEx for Detect2D {
    fn name(&self) -> Option<&ModuleName> {
        self.name.as_ref()
    }

    fn input_paths(&self) -> ModuleInput<'_> {
        self.from.as_ref().into()
    }

    fn output_shape(&self, input_shape: ShapeInput<'_>) -> Option<ShapeOutput> {
        let Self {
            classes,
            ref anchors,
            ..
        } = *self;
        let input_shape = input_shape.single_tensor()?;

        match input_shape.as_ref() {
            &[_b, Dim::Size(c), _h, _w] => {
                let expect_c = anchors.len() * (1 + 4 + classes);
                if c == expect_c {
                    Some(ShapeOutput::Detect2D)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}
