use super::*;
use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UpSample2D {
    pub name: Option<ModuleName>,
    pub from: Option<ModulePath>,
    pub config: UpSample2DConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum UpSample2DConfig {
    ByScale { scale: R64 },
    ByStride { stride: usize, reverse: bool },
}

impl ModuleEx for UpSample2D {
    fn name(&self) -> Option<&ModuleName> {
        self.name.as_ref()
    }

    fn input_paths(&self) -> ModuleInput<'_> {
        self.from.as_ref().into()
    }

    fn output_shape(&self, input_shape: ShapeInput<'_>) -> Option<ShapeOutput> {
        match self.config {
            UpSample2DConfig::ByScale { scale } => match input_shape.single_tensor()?.as_ref() {
                &[b, c, h, w] => {
                    let out_h = h.scale(scale);
                    let out_w = w.scale(scale);
                    Some(vec![b, c, out_h, out_w].into())
                }
                _ => None,
            },
            UpSample2DConfig::ByStride { .. } => {
                todo!();
            }
        }
    }
}
