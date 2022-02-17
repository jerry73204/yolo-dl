use super::*;
use crate::common::*;
use tensor_shape::Shape;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "RawConv2D", into = "RawConv2D")]
pub struct Conv2D {
    pub name: Option<ModuleName>,
    pub from: Option<ModulePath>,
    pub c: usize,
    pub k: usize,
    pub s: usize,
    pub p: usize,
    pub d: usize,
    pub g: usize,
    pub bias: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct RawConv2D {
    pub name: Option<ModuleName>,
    pub from: Option<ModulePath>,
    pub c: usize,
    pub k: usize,
    #[serde(default = "default_stride")]
    pub s: usize,
    pub p: Option<usize>,
    #[serde(default = "default_dilation")]
    pub d: usize,
    #[serde(default = "default_group")]
    pub g: usize,
    #[serde(default = "default_bias")]
    pub bias: bool,
}

impl From<RawConv2D> for Conv2D {
    fn from(raw: RawConv2D) -> Self {
        let RawConv2D {
            name,
            from,
            c,
            k,
            s,
            p,
            d,
            g,
            bias,
        } = raw;

        let p = p.unwrap_or_else(|| k / 2);

        Self {
            name,
            from,
            c,
            k,
            s,
            p,
            d,
            g,
            bias,
        }
    }
}

impl From<Conv2D> for RawConv2D {
    fn from(orig: Conv2D) -> Self {
        let Conv2D {
            name,
            from,
            c,
            k,
            s,
            p,
            d,
            g,
            bias,
        } = orig;

        Self {
            name,
            from,
            c,
            k,
            s,
            p: Some(p),
            d,
            g,
            bias,
        }
    }
}

impl Conv2D {
    pub fn new(
        name: impl Into<Option<ModuleName>>,
        from: impl Into<Option<ModulePath>>,
        c: usize,
        k: usize,
    ) -> Self {
        Self {
            name: name.into(),
            from: from.into(),
            c,
            k,
            s: default_stride(),
            p: k / 2,
            d: 1,
            g: default_group(),
            bias: default_bias(),
        }
    }
}

impl ModuleEx for Conv2D {
    fn name(&self) -> Option<&ModuleName> {
        self.name.as_ref()
    }

    fn input_paths(&self) -> ModuleInput<'_> {
        self.from.as_ref().into()
    }

    fn output_shape(&self, input_shape: ShapeInput<'_>) -> Option<ShapeOutput> {
        let Self {
            c: out_c,
            k,
            s,
            p,
            d,
            ..
        } = *self;

        let [in_b, _in_c, in_h, in_w] = match input_shape.single_tensor()?.as_ref() {
            &[b, c, h, w] => [b, c, h, w],
            _ => return None,
        };

        let out_h = (in_h + 2 * p - d * (k - 1) - 1) / s + 1;
        let out_w = (in_w + 2 * p - d * (k - 1) - 1) / s + 1;
        let output_shape: Shape = vec![in_b, out_c.into(), out_h, out_w].into();

        Some(output_shape.into())
    }
}

fn default_stride() -> usize {
    1
}

fn default_dilation() -> usize {
    1
}

fn default_group() -> usize {
    1
}

fn default_bias() -> bool {
    true
}
