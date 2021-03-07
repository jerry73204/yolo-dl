use super::*;
use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SppCsp2D {
    pub name: Option<ModuleName>,
    pub from: Option<ModulePath>,
    pub c: usize,
    pub k: Vec<usize>,
    #[serde(default = "default_c_mul")]
    pub c_mul: R64,
    #[serde(default)]
    pub bn: BatchNorm,
}

impl SppCsp2D {
    pub fn new(name: Option<ModuleName>, from: Option<ModulePath>, c: usize) -> Self {
        Self {
            name,
            from,
            c,
            k: vec![1, 5, 9, 13],
            c_mul: default_c_mul(),
            bn: Default::default(),
        }
    }
}

impl ModuleEx for SppCsp2D {
    fn name(&self) -> Option<&ModuleName> {
        self.name.as_ref()
    }

    fn input_paths(&self) -> ModuleInput<'_> {
        self.from.as_ref().into()
    }

    fn output_shape(&self, input_shape: ShapeInput<'_>) -> Option<ShapeOutput> {
        let Self { c: out_c, .. } = *self;

        match input_shape.tensor()?.as_ref() {
            &[b, _c, h, w] => Some(vec![b, out_c.into(), h, w].into()),
            _ => None,
        }
    }
}

fn default_c_mul() -> R64 {
    r64(0.5)
}
