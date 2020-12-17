use super::{
    misc::Size,
    model::{GroupName, LayerEx, LayerName, LayerPath},
};
use crate::{common::*, utils};

pub use bottleneck::*;
pub use bottleneck_csp::*;
pub use concat::*;
pub use conv_block::*;
pub use detect::*;
pub use focus::*;
pub use group::*;
pub use input::*;
pub use output::*;
pub use spp::*;
pub use up_sample::*;

mod input {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Input {
        pub shape: Vec<usize>,
    }

    impl LayerEx for Input {
        fn input_layers(&self) -> Vec<&LayerPath> {
            [].as_ref().into()
        }
    }
}

mod output {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Output {
        pub from: Option<LayerPath>,
    }

    impl LayerEx for Output {
        fn input_layers(&self) -> Vec<&LayerPath> {
            self.from.as_ref().map(|path| vec![path]).unwrap_or(vec![])
        }
    }
}

mod conv_block {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct ConvBlock {
        pub from: Option<LayerPath>,
        pub out_c: usize,
        pub k: usize,
        pub s: usize,
        pub g: usize,
        pub with_activation: bool,
    }

    impl ConvBlock {
        pub fn new(from: Option<LayerPath>, out_c: usize) -> Self {
            Self {
                from,
                out_c,
                k: 1,
                s: 1,
                g: 1,
                with_activation: true,
            }
        }
    }

    impl LayerEx for ConvBlock {
        fn input_layers(&self) -> Vec<&LayerPath> {
            self.from.as_ref().map(|path| vec![path]).unwrap_or(vec![])
        }
    }
}

mod bottleneck {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Bottleneck {
        pub from: Option<LayerPath>,
        pub out_c: usize,
        pub shortcut: bool,
        pub g: usize,
        pub expansion: R64,
    }

    impl Bottleneck {
        pub fn new(from: Option<LayerPath>, out_c: usize) -> Self {
            Self {
                from,
                out_c,
                shortcut: true,
                g: 1,
                expansion: r64(0.5),
            }
        }
    }

    impl LayerEx for Bottleneck {
        fn input_layers(&self) -> Vec<&LayerPath> {
            self.from.as_ref().map(|path| vec![path]).unwrap_or(vec![])
        }
    }
}

mod bottleneck_csp {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct BottleneckCsp {
        pub from: Option<LayerPath>,
        pub out_c: usize,
        pub repeat: usize,
        pub shortcut: bool,
        pub g: usize,
        pub expansion: R64,
    }

    impl BottleneckCsp {
        pub fn new(from: Option<LayerPath>, out_c: usize) -> Self {
            Self {
                from,
                out_c,
                repeat: 1,
                shortcut: true,
                g: 1,
                expansion: r64(0.5),
            }
        }
    }

    impl LayerEx for BottleneckCsp {
        fn input_layers(&self) -> Vec<&LayerPath> {
            self.from.as_ref().map(|path| vec![path]).unwrap_or(vec![])
        }
    }
}

mod spp {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Spp {
        pub from: Option<LayerPath>,
        pub out_c: usize,
        pub ks: Vec<usize>,
    }

    impl Spp {
        pub fn new(from: Option<LayerPath>, out_c: usize) -> Self {
            Self {
                from,
                out_c,
                ks: vec![5, 9, 13],
            }
        }
    }

    impl LayerEx for Spp {
        fn input_layers(&self) -> Vec<&LayerPath> {
            self.from.as_ref().map(|path| vec![path]).unwrap_or(vec![])
        }
    }
}

mod focus {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Focus {
        pub from: Option<LayerPath>,
        pub out_c: usize,
        pub k: usize,
    }

    impl Focus {
        pub fn new(from: Option<LayerPath>, out_c: usize) -> Self {
            Self { from, out_c, k: 1 }
        }
    }

    impl LayerEx for Focus {
        fn input_layers(&self) -> Vec<&LayerPath> {
            self.from.as_ref().map(|path| vec![path]).unwrap_or(vec![])
        }
    }
}

mod detect {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Detect {
        pub from: Option<LayerPath>,
        pub num_classes: usize,
        pub anchors: Vec<Size>,
    }

    impl LayerEx for Detect {
        fn input_layers(&self) -> Vec<&LayerPath> {
            self.from.as_ref().map(|path| vec![path]).unwrap_or(vec![])
        }
    }
}

mod up_sample {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct UpSample {
        pub from: Option<LayerPath>,
        pub scale: R64,
    }

    impl LayerEx for UpSample {
        fn input_layers(&self) -> Vec<&LayerPath> {
            self.from.as_ref().map(|path| vec![path]).unwrap_or(vec![])
        }
    }
}

mod concat {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    pub struct Concat {
        #[derivative(Hash(hash_with = "utils::hash_vec_indexset::<LayerPath, _>"))]
        pub from: IndexSet<LayerPath>,
    }

    impl LayerEx for Concat {
        fn input_layers(&self) -> Vec<&LayerPath> {
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
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<LayerName, LayerPath, _>"))]
        pub from: IndexMap<LayerName, LayerPath>,
        pub group: GroupName,
    }

    impl LayerEx for Group {
        fn input_layers(&self) -> Vec<&LayerPath> {
            let from: Vec<_> = self.from.values().collect();
            from
        }
    }
}
