use super::*;
use crate::{common::*, config::misc::Shape};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Concat2D {
    pub name: Option<ModuleName>,
    pub from: Vec<ModulePath>,
}

impl ModuleEx for Concat2D {
    fn name(&self) -> Option<&ModuleName> {
        self.name.as_ref()
    }

    fn input_paths(&self) -> ModuleInput<'_> {
        self.from.as_slice().into()
    }

    fn output_shape(&self, input_shape: ShapeInput<'_>) -> Option<ShapeOutput> {
        let Self { from, .. } = self;
        let input_shapes = input_shape.indexed_tensor()?;

        if input_shapes.len() != from.len() {
            return None;
        }

        let acc = match input_shapes[0].as_ref() {
            &[b, c, h, w] => [b, c, h, w],
            _ => return None,
        };

        let output_shape =
            input_shapes[1..]
                .iter()
                .try_fold(acc, |acc, in_shape| match in_shape.as_ref() {
                    &[b, c, h, w] => {
                        let [acc_b, acc_c, acc_h, acc_w] = acc;
                        Some([
                            acc_b.equalize(&b)?,
                            acc_c + c,
                            acc_h.equalize(&h)?,
                            acc_w.equalize(&w)?,
                        ])
                    }
                    _ => None,
                })?;
        let output_shape: Shape = Vec::from(output_shape).into();

        Some(output_shape.into())
    }
}
