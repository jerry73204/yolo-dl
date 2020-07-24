use super::*;
use crate::common::*;

pub fn yolo_v5_small_init(input_channels: usize, num_classes: usize) -> YoloInit {
    YoloInit {
        input_channels,
        num_classes,
        depth_multiple: R64::new(0.33),
        width_multiple: R64::new(0.50),
        layers: vec![
            // backbone
            LayerInit {
                name: Some("backbone-p1".into()),
                kind: LayerKind::Focus {
                    from: None,
                    out_c: 64,
                    k: 3,
                },
            },
            LayerInit {
                name: Some("backbone-p2".into()),
                kind: LayerKind::ConvBlock {
                    from: None,
                    out_c: 128,
                    k: 3,
                    s: 2,
                },
            },
            LayerInit {
                name: None,
                kind: LayerKind::BottleneckCsp {
                    from: None,
                    repeat: 3,
                    shortcut: true,
                },
            },
            LayerInit {
                name: Some("backbone-p3".into()),
                kind: LayerKind::ConvBlock {
                    from: None,
                    out_c: 256,
                    k: 3,
                    s: 2,
                },
            },
            LayerInit {
                name: None,
                kind: LayerKind::BottleneckCsp {
                    from: None,
                    repeat: 9,
                    shortcut: true,
                },
            },
            LayerInit {
                name: Some("backbone-p4".into()),
                kind: LayerKind::ConvBlock {
                    from: None,
                    out_c: 512,
                    k: 3,
                    s: 2,
                },
            },
            LayerInit {
                name: None,
                kind: LayerKind::BottleneckCsp {
                    from: None,
                    repeat: 9,
                    shortcut: true,
                },
            },
            LayerInit {
                name: Some("backbone-p5".into()),
                kind: LayerKind::ConvBlock {
                    from: None,
                    out_c: 1024,
                    k: 3,
                    s: 2,
                },
            },
            LayerInit {
                name: None,
                kind: LayerKind::Spp {
                    from: None,
                    out_c: 1024,
                    ks: vec![5, 9, 13],
                },
            },
            // head p5
            LayerInit {
                name: Some("head-p5".into()),
                kind: LayerKind::BottleneckCsp {
                    from: None,
                    repeat: 3,
                    shortcut: false,
                },
            },
            // head p4
            LayerInit {
                name: None,
                kind: LayerKind::ConvBlock {
                    from: None,
                    out_c: 512,
                    k: 1,
                    s: 1,
                },
            },
            LayerInit {
                name: Some("upsample-p4".into()),
                kind: LayerKind::Upsample {
                    from: None,
                    scale_factor: R64::new(2.0),
                },
            },
            LayerInit {
                name: None,
                kind: LayerKind::Concat {
                    from: vec!["backbone-p4".into(), "upsample-p4".into()],
                },
            },
            LayerInit {
                name: Some("head-p4".into()),
                kind: LayerKind::BottleneckCsp {
                    from: None,
                    repeat: 3,
                    shortcut: false,
                },
            },
            // head p3
            LayerInit {
                name: None,
                kind: LayerKind::ConvBlock {
                    from: None,
                    out_c: 256,
                    k: 1,
                    s: 1,
                },
            },
            LayerInit {
                name: Some("upsample-p3".into()),
                kind: LayerKind::Upsample {
                    from: None,
                    scale_factor: R64::new(2.0),
                },
            },
            LayerInit {
                name: None,
                kind: LayerKind::Concat {
                    from: vec!["backbone-p3".into(), "upsample-p3".into()],
                },
            },
            LayerInit {
                name: None,
                kind: LayerKind::BottleneckCsp {
                    from: None,
                    repeat: 3,
                    shortcut: false,
                },
            },
            LayerInit {
                name: None,
                kind: LayerKind::HeadConv2d {
                    from: None,
                    k: 1,
                    s: 1,
                    anchors: vec![(116, 90), (156, 198), (373, 326)], // P5/32
                },
            },
            // head p2
            LayerInit {
                name: Some("head-conv-p2".into()),
                kind: LayerKind::ConvBlock {
                    from: None,
                    out_c: 256,
                    k: 3,
                    s: 2,
                },
            },
            LayerInit {
                name: None,
                kind: LayerKind::Concat {
                    from: vec!["head-conv-p2".into(), "head-p4".into()],
                },
            },
            LayerInit {
                name: None,
                kind: LayerKind::BottleneckCsp {
                    from: None,
                    repeat: 3,
                    shortcut: false,
                },
            },
            LayerInit {
                name: None,
                kind: LayerKind::HeadConv2d {
                    from: None,
                    k: 1,
                    s: 1,
                    anchors: vec![(30, 61), (62, 45), (59, 119)], // P4/1/6
                },
            },
            // head p1
            LayerInit {
                name: Some("head-conv-p1".into()),
                kind: LayerKind::ConvBlock {
                    from: None,
                    out_c: 512,
                    k: 3,
                    s: 2,
                },
            },
            LayerInit {
                name: None,
                kind: LayerKind::Concat {
                    from: vec!["head-conv-p1".into(), "head-p5".into()],
                },
            },
            LayerInit {
                name: None,
                kind: LayerKind::BottleneckCsp {
                    from: None,
                    repeat: 3,
                    shortcut: false,
                },
            },
            LayerInit {
                name: None,
                kind: LayerKind::HeadConv2d {
                    from: None,
                    k: 1,
                    s: 1,
                    anchors: vec![(10, 13), (16, 30), (33, 23)], // P3/8
                },
            },
        ],
    }
}

pub fn yolo_v5_small<'p, P>(path: P, input_channels: usize, num_classes: usize) -> YoloModel
where
    P: Borrow<nn::Path<'p>>,
{
    let init = yolo_v5_small_init(input_channels, num_classes);
    let model = init.build(path).unwrap();
    model
}
