use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Yolo {
    pub input_channels: usize,
    pub num_classes: usize,
    pub depth_multiple: R64,
    pub width_multiple: R64,
    pub layers: Vec<Layer>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Layer {
    pub name: Option<String>,
    pub kind: LayerKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum LayerKind {
    Focus {
        from: Option<String>,
        out_c: usize,
        k: usize,
    },
    ConvBlock {
        from: Option<String>,
        out_c: usize,
        k: usize,
        s: usize,
    },
    Bottleneck {
        from: Option<String>,
        repeat: usize,
    },
    BottleneckCsp {
        from: Option<String>,
        repeat: usize,
        shortcut: bool,
    },
    Spp {
        from: Option<String>,
        out_c: usize,
        ks: Vec<usize>,
    },
    HeadConv2d {
        from: Option<String>,
        k: usize,
        s: usize,
        anchors: Vec<(usize, usize)>,
    },
    Upsample {
        from: Option<String>,
        scale_factor: R64,
    },
    Concat {
        from: Vec<String>,
    },
}

impl LayerKind {
    pub fn from_name(&self) -> Option<&str> {
        match self {
            Self::Focus { from, .. } => from.as_ref().map(|name| name.as_str()),
            Self::ConvBlock { from, .. } => from.as_ref().map(|name| name.as_str()),
            Self::Bottleneck { from, .. } => from.as_ref().map(|name| name.as_str()),
            Self::BottleneckCsp { from, .. } => from.as_ref().map(|name| name.as_str()),
            Self::Spp { from, .. } => from.as_ref().map(|name| name.as_str()),
            Self::HeadConv2d { from, .. } => from.as_ref().map(|name| name.as_str()),
            Self::Upsample { from, .. } => from.as_ref().map(|name| name.as_str()),
            _ => None,
        }
    }

    pub fn from_multiple_names(&self) -> Option<&[String]> {
        let names = match self {
            Self::Concat { from, .. } => from.as_slice(),
            _ => return None,
        };
        Some(names)
    }
}
