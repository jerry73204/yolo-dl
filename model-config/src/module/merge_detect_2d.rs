use super::*;
use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MergeDetect2D {
    pub name: Option<ModuleName>,
    pub from: Vec<ModulePath>,
}

impl ModuleEx for MergeDetect2D {
    fn name(&self) -> Option<&ModuleName> {
        self.name.as_ref()
    }

    fn input_paths(&self) -> ModuleInput<'_> {
        self.from.as_slice().into()
    }

    fn output_shape(&self, input_shape: ShapeInput<'_>) -> Option<ShapeOutput> {
        if input_shape.is_indexed_detect_2d() {
            Some(ShapeOutput::MergeDetect2D)
        } else {
            None
        }
    }
}
