use super::*;
use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UpSample2D {
    pub name: Option<ModuleName>,
    pub from: Option<ModulePath>,
    pub scale: R64,
}

impl ModuleEx for UpSample2D {
    fn name(&self) -> Option<&ModuleName> {
        self.name.as_ref()
    }

    fn input_paths(&self) -> ModuleInput<'_> {
        self.from.as_ref().into()
    }

    fn output_shape(&self, input_shape: ShapeInput<'_>) -> Option<ShapeOutput> {
        let Self { scale, .. } = *self;

        match input_shape.tensor()?.as_ref() {
            &[b, c, h, w] => {
                let out_h = h.scale_r64(scale);
                let out_w = w.scale_r64(scale);
                Some(vec![b, c, out_h, out_w].into())
            }
            _ => None,
        }
    }
}
