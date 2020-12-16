use crate::{common::*, misc::Size, model::LayerEx, utils};

pub use bottleneck::*;
pub use bottleneck_csp::*;
pub use concat::*;
pub use conv_block::*;
pub use detect::*;
pub use focus::*;
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
        fn input_names(&self) -> Cow<'_, [&str]> {
            [].as_ref().into()
        }
    }
}

mod output {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Output {
        pub from: Option<String>,
    }

    impl LayerEx for Output {
        fn input_names(&self) -> Cow<'_, [&str]> {
            self.from
                .as_ref()
                .map(|name| vec![name.as_str()])
                .unwrap_or(vec![])
                .into()
        }
    }
}

mod conv_block {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct ConvBlock {
        pub from: Option<String>,
        pub out_c: usize,
        pub k: usize,
        pub s: usize,
        pub g: usize,
        pub with_activation: bool,
    }

    impl ConvBlock {
        pub fn new(from: Option<String>, out_c: usize) -> Self {
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
        fn input_names(&self) -> Cow<'_, [&str]> {
            self.from
                .as_ref()
                .map(|name| vec![name.as_str()])
                .unwrap_or(vec![])
                .into()
        }
    }
}

mod bottleneck {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Bottleneck {
        pub from: Option<String>,
        pub out_c: usize,
        pub shortcut: bool,
        pub g: usize,
        pub expansion: R64,
    }

    impl Bottleneck {
        pub fn new(from: Option<String>, out_c: usize) -> Self {
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
        fn input_names(&self) -> Cow<'_, [&str]> {
            self.from
                .as_ref()
                .map(|name| vec![name.as_str()])
                .unwrap_or(vec![])
                .into()
        }
    }
}

mod bottleneck_csp {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct BottleneckCsp {
        pub from: Option<String>,
        pub out_c: usize,
        pub repeat: usize,
        pub shortcut: bool,
        pub g: usize,
        pub expansion: R64,
    }

    impl BottleneckCsp {
        pub fn new(from: Option<String>, out_c: usize) -> Self {
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
        fn input_names(&self) -> Cow<'_, [&str]> {
            self.from
                .as_ref()
                .map(|name| vec![name.as_str()])
                .unwrap_or(vec![])
                .into()
        }
    }
}

mod spp {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Spp {
        pub from: Option<String>,
        pub out_c: usize,
        pub ks: Vec<usize>,
    }

    impl Spp {
        pub fn new(from: Option<String>, out_c: usize) -> Self {
            Self {
                from,
                out_c,
                ks: vec![5, 9, 13],
            }
        }
    }

    impl LayerEx for Spp {
        fn input_names(&self) -> Cow<'_, [&str]> {
            self.from
                .as_ref()
                .map(|name| vec![name.as_str()])
                .unwrap_or(vec![])
                .into()
        }
    }
}

mod focus {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Focus {
        pub from: Option<String>,
        pub out_c: usize,
        pub k: usize,
    }

    impl Focus {
        pub fn new(from: Option<String>, out_c: usize) -> Self {
            Self { from, out_c, k: 1 }
        }
    }

    impl LayerEx for Focus {
        fn input_names(&self) -> Cow<'_, [&str]> {
            self.from
                .as_ref()
                .map(|name| vec![name.as_str()])
                .unwrap_or(vec![])
                .into()
        }
    }
}

mod detect {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Detect {
        pub from: Option<String>,
        pub num_classes: usize,
        pub anchors: Vec<Size>,
    }

    impl LayerEx for Detect {
        fn input_names(&self) -> Cow<'_, [&str]> {
            self.from
                .as_ref()
                .map(|name| vec![name.as_str()])
                .unwrap_or(vec![])
                .into()
        }
    }
}

mod up_sample {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct UpSample {
        pub from: Option<String>,
        pub scale: R64,
    }

    impl LayerEx for UpSample {
        fn input_names(&self) -> Cow<'_, [&str]> {
            self.from
                .as_ref()
                .map(|name| vec![name.as_str()])
                .unwrap_or(vec![])
                .into()
        }
    }
}

mod concat {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    pub struct Concat {
        #[derivative(Hash(hash_with = "utils::hash_vec_indexset::<String, _>"))]
        pub from: IndexSet<String>,
    }

    impl LayerEx for Concat {
        fn input_names(&self) -> Cow<'_, [&str]> {
            let from: Vec<_> = self.from.iter().map(|name| name.as_str()).collect();
            from.into()
        }
    }
}
