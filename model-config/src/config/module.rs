use super::{
    group::GroupName,
    misc::{Activation, Dim, Shape, Size},
};
use crate::{common::*, utils};

pub use concat_2d::*;
pub use conv_bn_2d_block::*;
pub use dark_csp_2d::*;
pub use detect_2d::*;
pub use group_ref::*;
pub use input::*;
pub use input_path::*;
pub use merge_detect_2d::*;
pub use module::*;
pub use module_input::*;
pub use module_name::*;
pub use shape_input::*;
pub use shape_output::*;
pub use spp_csp_2d::*;
pub use sum_2d::*;
pub use up_sample_2d::*;

mod module_input {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Derivative)]
    #[derivative(Hash)]
    pub enum ModuleInput<'a> {
        None,
        PlaceHolder,
        Infer,
        Path(&'a ModulePath),
        Indexed(&'a [ModulePath]),
        Named(
            #[derivative(Hash(
                hash_with = "utils::hash_vec_indexmap::<ModuleName, ModulePath, _>"
            ))]
            &'a IndexMap<ModuleName, ModulePath>,
        ),
    }

    impl<'a> From<&'a ModulePath> for ModuleInput<'a> {
        fn from(from: &'a ModulePath) -> Self {
            Self::Path(from)
        }
    }

    impl<'a> From<&'a Option<ModulePath>> for ModuleInput<'a> {
        fn from(from: &'a Option<ModulePath>) -> Self {
            from.as_ref().into()
        }
    }

    impl<'a> From<Option<&'a ModulePath>> for ModuleInput<'a> {
        fn from(from: Option<&'a ModulePath>) -> Self {
            match from {
                Some(path) => path.into(),
                None => ModuleInput::Infer,
            }
        }
    }

    impl<'a> From<&'a [ModulePath]> for ModuleInput<'a> {
        fn from(from: &'a [ModulePath]) -> Self {
            Self::Indexed(from)
        }
    }

    impl<'a> From<&'a IndexMap<ModuleName, ModulePath>> for ModuleInput<'a> {
        fn from(from: &'a IndexMap<ModuleName, ModulePath>) -> Self {
            Self::Named(from)
        }
    }
}

mod shape_input {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Derivative)]
    #[derivative(Hash)]
    pub enum ShapeKind<'a> {
        Tensor(&'a Shape),
        Detect2D,
    }

    impl ShapeKind<'_> {
        pub fn tensor(&self) -> Option<&Shape> {
            match self {
                Self::Tensor(shape) => Some(shape),
                _ => None,
            }
        }

        pub fn is_detect_2d(&self) -> bool {
            match self {
                Self::Detect2D => true,
                _ => false,
            }
        }
    }

    impl<'a> From<&'a Shape> for ShapeKind<'a> {
        fn from(from: &'a Shape) -> Self {
            Self::Tensor(from)
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Derivative)]
    #[derivative(Hash)]
    pub enum ShapeInput<'a> {
        None,
        PlaceHolder,
        Single(ShapeKind<'a>),
        Indexed(Vec<ShapeKind<'a>>),
    }

    impl ShapeInput<'_> {
        pub fn is_none(&self) -> bool {
            match self {
                Self::None => true,
                _ => false,
            }
        }

        pub fn is_placeholder(&self) -> bool {
            match self {
                Self::PlaceHolder => true,
                _ => false,
            }
        }

        pub fn tensor(&self) -> Option<&Shape> {
            match self {
                Self::Single(ShapeKind::Tensor(shape)) => Some(shape),
                _ => None,
            }
        }

        pub fn is_detect_2d(&self) -> bool {
            match self {
                Self::Single(ShapeKind::Detect2D) => true,
                _ => false,
            }
        }

        pub fn indexed_tensor(&self) -> Option<Vec<&Shape>> {
            match self {
                Self::Indexed(vec) => {
                    let shapes: Option<Vec<_>> = vec.iter().map(|kind| kind.tensor()).collect();
                    shapes
                }
                _ => None,
            }
        }

        pub fn is_indexed_detect_2d(&self) -> bool {
            match self {
                Self::Indexed(vec) => vec.iter().all(|kind| kind.is_detect_2d()),
                _ => false,
            }
        }
    }

    impl<'a> From<&'a Shape> for ShapeInput<'a> {
        fn from(from: &'a Shape) -> Self {
            Self::Single(from.into())
        }
    }

    impl<'a> TryFrom<&'a ShapeOutput> for ShapeInput<'a> {
        type Error = Error;

        fn try_from(from: &'a ShapeOutput) -> Result<Self, Self::Error> {
            let input_shape: Self = match from {
                ShapeOutput::Shape(shape) => shape.into(),
                ShapeOutput::Detect2D => Self::Single(ShapeKind::Detect2D),
                ShapeOutput::MergeDetect2D => bail!("TODO"),
            };
            Ok(input_shape)
        }
    }

    impl<'a, 'b> From<&'b [&'a Shape]> for ShapeInput<'a> {
        fn from(from: &'b [&'a Shape]) -> Self {
            let shapes: Vec<ShapeKind> = from.iter().cloned().map(|shape| shape.into()).collect();
            Self::Indexed(shapes)
        }
    }

    impl<'a, 'b> TryFrom<&'b [&'a ShapeOutput]> for ShapeInput<'a> {
        type Error = Error;

        fn try_from(from: &'b [&'a ShapeOutput]) -> Result<Self, Self::Error> {
            let kinds: Vec<_> = from
                .iter()
                .map(|output_shape| -> Result<_> {
                    let kind = match output_shape {
                        ShapeOutput::Shape(shape) => ShapeKind::Tensor(shape),
                        ShapeOutput::Detect2D => ShapeKind::Detect2D,
                        _ => bail!("TODO"),
                    };
                    Ok(kind)
                })
                .try_collect()?;
            Ok(Self::Indexed(kinds))
        }
    }
}

mod shape_output {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    pub enum ShapeOutput {
        Shape(Shape),
        Detect2D,
        MergeDetect2D,
    }

    impl ShapeOutput {
        pub fn tensor(self) -> Option<Shape> {
            match self {
                Self::Shape(shape) => Some(shape),
                _ => None,
            }
        }

        pub fn as_tensor(&self) -> Option<&Shape> {
            match self {
                Self::Shape(shape) => Some(shape),
                _ => None,
            }
        }

        pub fn is_detect_2d(&self) -> bool {
            match self {
                Self::Detect2D => true,
                _ => false,
            }
        }

        pub fn is_merge_detect_2d(&self) -> bool {
            match self {
                Self::MergeDetect2D => true,
                _ => false,
            }
        }
    }

    impl From<Shape> for ShapeOutput {
        fn from(from: Shape) -> Self {
            Self::Shape(from)
        }
    }

    impl From<Vec<Dim>> for ShapeOutput {
        fn from(from: Vec<Dim>) -> Self {
            Self::Shape(Shape::from(from))
        }
    }

    impl Display for ShapeOutput {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            match self {
                Self::Shape(shape) => Display::fmt(shape, f),
                Self::Detect2D => write!(f, "Detect2D"),
                Self::MergeDetect2D => write!(f, "MergeDetect2D"),
            }
        }
    }
}

mod input_path {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct GroupPath {
        pub layer: ModuleName,
        pub output: ModuleName,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct ModulePath(Vec<ModuleName>);

    impl ModulePath {
        pub fn empty() -> Self {
            Self(vec![])
        }

        pub fn is_empty(&self) -> bool {
            self.0.is_empty()
        }

        pub fn depth(&self) -> usize {
            self.0.len()
        }

        pub fn extend(&self, other: &ModulePath) -> Self {
            self.0.iter().chain(other.0.iter()).collect()
        }

        pub fn join<'a>(&self, name: impl Into<Cow<'a, ModuleName>>) -> Self {
            let name = name.into().into_owned();
            self.0.iter().cloned().chain(iter::once(name)).collect()
        }
    }

    impl FromStr for ModulePath {
        type Err = Error;

        fn from_str(name: &str) -> Result<Self, Self::Err> {
            let tokens = name.split('.');
            let components: Vec<_> = tokens
                .map(|token| ModuleName::from_str(token))
                .try_collect()?;
            Ok(Self(components))
        }
    }

    impl Serialize for ModulePath {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            self.to_string().serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for ModulePath {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let text = String::deserialize(deserializer)?;
            let ident = Self::from_str(&text)
                .map_err(|err| D::Error::custom(format!("invalid name: {:?}", err)))?;
            Ok(ident)
        }
    }

    impl Display for ModulePath {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            let text = self.0.iter().map(AsRef::as_ref).join(".");
            Display::fmt(&text, f)
        }
    }

    impl AsRef<[ModuleName]> for ModulePath {
        fn as_ref(&self) -> &[ModuleName] {
            self.0.as_ref()
        }
    }

    impl FromIterator<ModuleName> for ModulePath {
        fn from_iter<T>(iter: T) -> Self
        where
            T: IntoIterator<Item = ModuleName>,
        {
            Self(Vec::from_iter(iter))
        }
    }

    impl<'a> FromIterator<&'a ModuleName> for ModulePath {
        fn from_iter<T>(iter: T) -> Self
        where
            T: IntoIterator<Item = &'a ModuleName>,
        {
            Self(iter.into_iter().cloned().collect())
        }
    }
}

mod module_name {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct ModuleName(String);

    impl FromStr for ModuleName {
        type Err = Error;

        fn from_str(name: &str) -> Result<Self> {
            ensure!(!name.is_empty(), "module name must not be empty");
            ensure!(!name.contains('.'), "module name must not contain dot '.'");
            Ok(Self(name.to_owned()))
        }
    }

    impl Serialize for ModuleName {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            self.0.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for ModuleName {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let text = String::deserialize(deserializer)?;
            let name = Self::from_str(&text)
                .map_err(|err| D::Error::custom(format!("invalid name: {:?}", err)))?;
            Ok(name)
        }
    }

    impl Borrow<str> for ModuleName {
        fn borrow(&self) -> &str {
            self.0.as_ref()
        }
    }

    impl AsRef<str> for ModuleName {
        fn as_ref(&self) -> &str {
            self.0.as_ref()
        }
    }

    impl Display for ModuleName {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            Display::fmt(&self.0, f)
        }
    }

    impl From<ModuleName> for Cow<'_, ModuleName> {
        fn from(from: ModuleName) -> Self {
            Cow::Owned(from)
        }
    }

    impl<'a> From<&'a ModuleName> for Cow<'a, ModuleName> {
        fn from(from: &'a ModuleName) -> Self {
            Cow::Borrowed(from)
        }
    }
}

mod module {
    use super::*;

    pub trait ModuleEx {
        fn name(&self) -> Option<&ModuleName>;
        fn input_paths(&self) -> ModuleInput<'_>;
        fn output_shape(&self, input_shape: ShapeInput<'_>) -> Option<ShapeOutput>;
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, AsRefStr, Serialize, Deserialize)]
    #[serde(tag = "kind")]
    pub enum Module {
        Input(Input),
        // Focus(Focus),
        ConvBn2D(ConvBn2D),
        // Bottleneck(Bottleneck),
        // BottleneckCsp(BottleneckCsp),
        // Spp(Spp),
        DarkCsp2D(DarkCsp2D),
        SppCsp2D(SppCsp2D),
        UpSample2D(UpSample2D),
        Concat2D(Concat2D),
        Sum2D(Sum2D),
        Detect2D(Detect2D),
        GroupRef(GroupRef),
        MergeDetect2D(MergeDetect2D),
    }

    impl ModuleEx for Module {
        fn name(&self) -> Option<&ModuleName> {
            match self {
                Module::Input(layer) => layer.name(),
                Module::ConvBn2D(layer) => layer.name(),
                Module::UpSample2D(layer) => layer.name(),
                Module::DarkCsp2D(layer) => layer.name(),
                Module::SppCsp2D(layer) => layer.name(),
                Module::Concat2D(layer) => layer.name(),
                Module::Sum2D(layer) => layer.name(),
                Module::Detect2D(layer) => layer.name(),
                Module::GroupRef(layer) => layer.name(),
                Module::MergeDetect2D(layer) => layer.name(),
            }
        }

        fn input_paths(&self) -> ModuleInput<'_> {
            match self {
                Module::Input(layer) => layer.input_paths(),
                Module::ConvBn2D(layer) => layer.input_paths(),
                Module::UpSample2D(layer) => layer.input_paths(),
                Module::DarkCsp2D(layer) => layer.input_paths(),
                Module::SppCsp2D(layer) => layer.input_paths(),
                Module::Concat2D(layer) => layer.input_paths(),
                Module::Sum2D(layer) => layer.input_paths(),
                Module::Detect2D(layer) => layer.input_paths(),
                Module::GroupRef(layer) => layer.input_paths(),
                Module::MergeDetect2D(layer) => layer.input_paths(),
            }
        }

        fn output_shape(&self, input_shape: ShapeInput<'_>) -> Option<ShapeOutput> {
            match self {
                Module::Input(layer) => layer.output_shape(input_shape),
                Module::ConvBn2D(layer) => layer.output_shape(input_shape),
                Module::UpSample2D(layer) => layer.output_shape(input_shape),
                Module::DarkCsp2D(layer) => layer.output_shape(input_shape),
                Module::SppCsp2D(layer) => layer.output_shape(input_shape),
                Module::Concat2D(layer) => layer.output_shape(input_shape),
                Module::Sum2D(layer) => layer.output_shape(input_shape),
                Module::Detect2D(layer) => layer.output_shape(input_shape),
                Module::GroupRef(layer) => layer.output_shape(input_shape),
                Module::MergeDetect2D(layer) => layer.output_shape(input_shape),
            }
        }
    }
}

mod input {
    use super::*;

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
                ShapeInput::Single(ShapeKind::Tensor(input_shape)) => {
                    input_shape.equalize(&self.shape)?
                }
                ShapeInput::PlaceHolder => self.shape.clone(),
                _ => return None,
            };

            Some(output_shape.into())
        }
    }
}

mod conv_bn_2d_block {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(from = "RawConvBn2D", into = "RawConvBn2D")]
    pub struct ConvBn2D {
        pub name: Option<ModuleName>,
        pub from: Option<ModulePath>,
        pub c: usize,
        pub k: usize,
        pub s: usize,
        pub p: usize,
        pub d: usize,
        pub g: usize,
        pub act: Activation,
        pub bn: bool,
        pub bn_affine: bool,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    struct RawConvBn2D {
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
        #[serde(default = "default_activation")]
        pub act: Activation,
        #[serde(default = "default_batch_norm")]
        pub bn: bool,
        #[serde(default = "default_batch_norm_affine")]
        pub bn_affine: bool,
    }

    impl From<RawConvBn2D> for ConvBn2D {
        fn from(raw: RawConvBn2D) -> Self {
            let RawConvBn2D {
                name,
                from,
                c,
                k,
                s,
                p,
                d,
                g,
                act,
                bn,
                bn_affine,
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
                act,
                bn,
                bn_affine,
            }
        }
    }

    impl From<ConvBn2D> for RawConvBn2D {
        fn from(orig: ConvBn2D) -> Self {
            let ConvBn2D {
                name,
                from,
                c,
                k,
                s,
                p,
                d,
                g,
                act,
                bn,
                bn_affine,
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
                act,
                bn,
                bn_affine,
            }
        }
    }

    impl ConvBn2D {
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
                act: default_activation(),
                bn: default_batch_norm(),
                bn_affine: default_batch_norm_affine(),
            }
        }
    }

    impl ModuleEx for ConvBn2D {
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

            let [in_b, _in_c, in_h, in_w] = match input_shape.tensor()?.as_ref() {
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

    fn default_activation() -> Activation {
        Activation::Mish
    }

    fn default_batch_norm() -> bool {
        true
    }

    fn default_batch_norm_affine() -> bool {
        false
    }
}

mod dark_csp_2d {
    use super::*;

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
}

mod spp_csp_2d {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct SppCsp2D {
        pub name: Option<ModuleName>,
        pub from: Option<ModulePath>,
        pub c: usize,
        pub k: Vec<usize>,
        #[serde(default = "default_c_mul")]
        pub c_mul: R64,
        #[serde(default = "default_batch_norm")]
        pub bn: bool,
        #[serde(default = "default_batch_norm_affine")]
        pub bn_affine: bool,
    }

    impl SppCsp2D {
        pub fn new(name: Option<ModuleName>, from: Option<ModulePath>, c: usize) -> Self {
            Self {
                name,
                from,
                c,
                k: vec![1, 5, 9, 13],
                c_mul: default_c_mul(),
                bn: default_batch_norm(),
                bn_affine: default_batch_norm_affine(),
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

    fn default_batch_norm() -> bool {
        true
    }

    fn default_batch_norm_affine() -> bool {
        false
    }
}

mod detect_2d {
    use super::*;

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
            let input_shape = input_shape.tensor()?;

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
}

mod up_sample_2d {
    use super::*;

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
}

mod concat_2d {
    use super::*;

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

            let output_shape = input_shapes[1..].iter().try_fold(acc, |acc, in_shape| {
                match in_shape.as_ref() {
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
                }
            })?;
            let output_shape: Shape = Vec::from(output_shape).into();

            Some(output_shape.into())
        }
    }
}

mod sum_2d {
    use super::*;

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
            let input_shapes: Vec<&Shape> = input_shape.indexed_tensor()?;
            let mut iter = input_shapes.iter().cloned();
            let first = iter.next()?.to_owned();
            let output_shape = iter.try_fold(first, |acc, shape| acc.equalize(shape))?;
            Some(output_shape.into())
        }
    }
}

mod group_ref {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    pub struct GroupRef {
        pub name: ModuleName,
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<ModuleName, ModulePath, _>"))]
        pub from: IndexMap<ModuleName, ModulePath>,
        pub group: GroupName,
    }

    impl ModuleEx for GroupRef {
        fn name(&self) -> Option<&ModuleName> {
            Some(&self.name)
        }

        fn input_paths(&self) -> ModuleInput<'_> {
            (&self.from).into()
        }

        fn output_shape(&self, _input_shape: ShapeInput<'_>) -> Option<ShapeOutput> {
            None
        }
    }
}

mod merge_detect_2d {
    use super::*;

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
}
