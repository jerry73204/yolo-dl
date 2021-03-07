use super::*;
use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DarkCsp2D {
    pub name: Option<ModuleName>,
    pub from: Option<ModulePath>,
    pub c: usize,
    pub repeat: usize,
    #[serde(default = "default_shortcut")]
    pub shortcut: bool,
    #[serde(default = "default_c_mul")]
    pub c_mul: R64,
    #[serde(default = "default_batch_norm")]
    pub bn: bool,
    #[serde(default = "default_batch_norm_affine")]
    pub bn_affine: bool,
}

impl DarkCsp2D {
    pub fn new(
        name: Option<ModuleName>,
        from: Option<ModulePath>,
        c: usize,
        repeat: usize,
    ) -> Self {
        Self {
            name,
            from,
            c,
            repeat,
            shortcut: default_shortcut(),
            c_mul: default_c_mul(),
            bn: default_batch_norm(),
            bn_affine: default_batch_norm_affine(),
        }
    }
}

impl ModuleEx for DarkCsp2D {
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

fn default_shortcut() -> bool {
    true
}

fn default_c_mul() -> R64 {
    r64(1.0)
}

fn default_batch_norm() -> bool {
    true
}

fn default_batch_norm_affine() -> bool {
    false
}
