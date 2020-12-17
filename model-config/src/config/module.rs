use super::{
    misc::{Activation, Shape, Size},
    model::{GroupName, ModuleName, ModulePath},
};
use crate::{common::*, utils};

// pub use bottleneck::*;
// pub use bottleneck_csp::*;
pub use concat_2d::*;
pub use conv_bn_2d_block::*;
pub use detect_2d::*;
// pub use focus::*;
pub use group::*;
pub use input::*;
pub use module::*;
pub use output::*;
// pub use spp::*;
pub use dark_csp_2d::*;
pub use merge_detect_2d::*;
pub use spp_csp_2d::*;
pub use sum_2d::*;
pub use up_sample_2d::*;

mod module {
    use super::*;

    pub trait ModuleEx {
        fn input_paths(&self) -> Vec<&ModulePath>;
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(tag = "kind")]
    pub enum Module {
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
        MergeDetect2D(MergeDetect2D),
        Group(Group),
    }

    impl Module {
        pub fn output_shape(&self, _input_shape: &[usize]) -> Option<Shape> {
            // match &self.kind {
            //     LayerKind::Input(layer) => layer.output_shape(input_shape),
            //     LayerKind::Focus(layer) => layer.output_shape(input_shape),
            //     LayerKind::ConvBlock(layer) => layer.output_shape(input_shape),
            //     LayerKind::Bottleneck(layer) => layer.output_shape(input_shape),
            //     LayerKind::BottleneckCsp(layer) => layer.output_shape(input_shape),
            //     LayerKind::Spp(layer) => layer.output_shape(input_shape),
            //     LayerKind::UpSample(layer) => layer.output_shape(input_shape),
            //     LayerKind::Concat(layer) => layer.output_shape(input_shape),
            //     LayerKind::Detect(layer) => layer.output_shape(input_shape),
            // }
            todo!();
        }
    }

    impl ModuleEx for Module {
        fn input_paths(&self) -> Vec<&ModulePath> {
            match self {
                Module::ConvBn2D(layer) => layer.input_paths(),
                Module::UpSample2D(layer) => layer.input_paths(),
                Module::DarkCsp2D(layer) => layer.input_paths(),
                Module::SppCsp2D(layer) => layer.input_paths(),
                Module::Concat2D(layer) => layer.input_paths(),
                Module::Sum2D(layer) => layer.input_paths(),
                Module::Detect2D(layer) => layer.input_paths(),
                Module::MergeDetect2D(layer) => layer.input_paths(),
                Module::Group(layer) => layer.input_paths(),
            }
        }
    }
}

mod input {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Input {
        pub shape: Shape,
    }

    impl ModuleEx for Input {
        fn input_paths(&self) -> Vec<&ModulePath> {
            [].as_ref().into()
        }
    }
}

mod output {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Output {
        pub from: ModulePath,
    }

    impl ModuleEx for Output {
        fn input_paths(&self) -> Vec<&ModulePath> {
            vec![&self.from]
        }
    }
}

mod conv_bn_2d_block {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(from = "RawConvBn2D", into = "RawConvBn2D")]
    pub struct ConvBn2D {
        pub from: Option<ModulePath>,
        pub c: usize,
        pub k: usize,
        pub s: usize,
        pub p: usize,
        pub d: usize,
        pub g: usize,
        pub act: Activation,
        pub bn: bool,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    struct RawConvBn2D {
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
    }

    impl From<RawConvBn2D> for ConvBn2D {
        fn from(raw: RawConvBn2D) -> Self {
            let RawConvBn2D {
                from,
                c,
                k,
                s,
                p,
                d,
                g,
                act,
                bn,
            } = raw;

            let p = p.unwrap_or_else(|| k / 2);

            Self {
                from,
                c,
                k,
                s,
                p,
                d,
                g,
                act,
                bn,
            }
        }
    }

    impl From<ConvBn2D> for RawConvBn2D {
        fn from(orig: ConvBn2D) -> Self {
            let ConvBn2D {
                from,
                c,
                k,
                s,
                p,
                d,
                g,
                act,
                bn,
            } = orig;

            Self {
                from,
                c,
                k,
                s,
                p: Some(p),
                d,
                g,
                act,
                bn,
            }
        }
    }

    impl ConvBn2D {
        pub fn new(from: Option<ModulePath>, c: usize, k: usize) -> Self {
            Self {
                from,
                c,
                k,
                s: default_stride(),
                p: k / 2,
                d: 1,
                g: default_group(),
                act: default_activation(),
                bn: default_batch_norm(),
            }
        }
    }

    impl ModuleEx for ConvBn2D {
        fn input_paths(&self) -> Vec<&ModulePath> {
            self.from.as_ref().map(|path| vec![path]).unwrap_or(vec![])
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
}

// mod bottleneck {
//     use super::*;

//     #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
//     pub struct Bottleneck {
//         pub from: Option<ModulePath>,
//         pub c: usize,
//         pub shortcut: bool,
//         pub g: usize,
//         pub expansion: R64,
//     }

//     impl Bottleneck {
//         pub fn new(from: Option<ModulePath>, c: usize) -> Self {
//             Self {
//                 from,
//                 c,
//                 shortcut: true,
//                 g: 1,
//                 expansion: r64(0.5),
//             }
//         }
//     }

//     impl ModuleEx for Bottleneck {
//         fn input_paths(&self) -> Vec<&ModulePath> {
//             self.from.as_ref().map(|path| vec![path]).unwrap_or(vec![])
//         }
//     }
// }

mod dark_csp_2d {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct DarkCsp2D {
        pub from: Option<ModulePath>,
        pub c: usize,
        pub repeat: usize,
        #[serde(default = "default_shortcut")]
        pub shortcut: bool,
        #[serde(default = "default_c_mul")]
        pub c_mul: R64,
    }

    impl DarkCsp2D {
        pub fn new(from: Option<ModulePath>, c: usize, repeat: usize) -> Self {
            Self {
                from,
                c,
                repeat,
                shortcut: default_shortcut(),
                c_mul: default_c_mul(),
            }
        }
    }

    impl ModuleEx for DarkCsp2D {
        fn input_paths(&self) -> Vec<&ModulePath> {
            self.from.as_ref().map(|path| vec![path]).unwrap_or(vec![])
        }
    }

    fn default_shortcut() -> bool {
        true
    }

    fn default_c_mul() -> R64 {
        r64(1.0)
    }
}

mod spp_csp_2d {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct SppCsp2D {
        pub from: Option<ModulePath>,
        pub c: usize,
        pub k: Vec<usize>,
    }

    impl SppCsp2D {
        pub fn new(from: Option<ModulePath>, c: usize) -> Self {
            Self {
                from,
                c,
                k: vec![1, 5, 9, 13],
            }
        }
    }

    impl ModuleEx for SppCsp2D {
        fn input_paths(&self) -> Vec<&ModulePath> {
            self.from.as_ref().map(|path| vec![path]).unwrap_or(vec![])
        }
    }
}

mod focus {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Focus {
        pub from: Option<ModulePath>,
        pub c: usize,
        pub k: usize,
    }

    impl Focus {
        pub fn new(from: Option<ModulePath>, c: usize) -> Self {
            Self { from, c, k: 1 }
        }
    }

    impl ModuleEx for Focus {
        fn input_paths(&self) -> Vec<&ModulePath> {
            self.from.as_ref().map(|path| vec![path]).unwrap_or(vec![])
        }
    }
}

mod detect_2d {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Detect2D {
        pub from: Option<ModulePath>,
        pub classes: usize,
        pub anchors: Vec<Size>,
    }

    impl ModuleEx for Detect2D {
        fn input_paths(&self) -> Vec<&ModulePath> {
            self.from.as_ref().map(|path| vec![path]).unwrap_or(vec![])
        }
    }
}

mod merge_detect_2d {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct MergeDetect2D {
        pub from: Vec<ModulePath>,
    }

    impl ModuleEx for MergeDetect2D {
        fn input_paths(&self) -> Vec<&ModulePath> {
            let from: Vec<_> = self.from.iter().collect();
            from
        }
    }
}

mod up_sample_2d {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct UpSample2D {
        pub from: Option<ModulePath>,
        pub scale: R64,
    }

    impl ModuleEx for UpSample2D {
        fn input_paths(&self) -> Vec<&ModulePath> {
            self.from.as_ref().map(|path| vec![path]).unwrap_or(vec![])
        }
    }
}

mod concat_2d {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    pub struct Concat2D {
        #[derivative(Hash(hash_with = "utils::hash_vec_indexset::<ModulePath, _>"))]
        pub from: IndexSet<ModulePath>,
    }

    impl ModuleEx for Concat2D {
        fn input_paths(&self) -> Vec<&ModulePath> {
            let from: Vec<_> = self.from.iter().collect();
            from
        }
    }
}

mod sum_2d {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    pub struct Sum2D {
        #[derivative(Hash(hash_with = "utils::hash_vec_indexset::<ModulePath, _>"))]
        pub from: IndexSet<ModulePath>,
    }

    impl ModuleEx for Sum2D {
        fn input_paths(&self) -> Vec<&ModulePath> {
            let from: Vec<_> = self.from.iter().collect();
            from
        }
    }
}

mod group {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    pub struct Group {
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<ModuleName, ModulePath, _>"))]
        pub from: IndexMap<ModuleName, ModulePath>,
        pub group: GroupName,
    }

    impl ModuleEx for Group {
        fn input_paths(&self) -> Vec<&ModulePath> {
            let from: Vec<_> = self.from.values().collect();
            from
        }
    }
}
