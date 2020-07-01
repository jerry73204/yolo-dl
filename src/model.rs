use crate::common::*;
use layers::*;

pub fn yolo_v5_init() -> YoloInit {
    YoloInit {
        input_channels: 3,
        num_classes: 80,
        depth_multiple: 0.33,
        width_multiple: 0.50,
        layers: vec![
            // backbone
            LayerInit {
                name: Some("backbone-p1".into()),
                export: false,
                kind: LayerKind::Focus {
                    from: None,
                    out_c: 64,
                    k: 3,
                },
            },
            LayerInit {
                name: Some("backbone-p2".into()),
                export: false,
                kind: LayerKind::ConvBlock {
                    from: None,
                    out_c: 128,
                    k: 3,
                    s: 2,
                },
            },
            LayerInit {
                name: None,
                export: false,
                kind: LayerKind::BottleneckCsp {
                    from: None,
                    repeat: 3,
                    shortcut: true,
                },
            },
            LayerInit {
                name: Some("backbone-p3".into()),
                export: false,
                kind: LayerKind::ConvBlock {
                    from: None,
                    out_c: 256,
                    k: 3,
                    s: 2,
                },
            },
            LayerInit {
                name: None,
                export: false,
                kind: LayerKind::BottleneckCsp {
                    from: None,
                    repeat: 9,
                    shortcut: true,
                },
            },
            LayerInit {
                name: Some("backbone-p4".into()),
                export: false,
                kind: LayerKind::ConvBlock {
                    from: None,
                    out_c: 512,
                    k: 3,
                    s: 2,
                },
            },
            LayerInit {
                name: None,
                export: false,
                kind: LayerKind::BottleneckCsp {
                    from: None,
                    repeat: 9,
                    shortcut: true,
                },
            },
            LayerInit {
                name: Some("backbone-p5".into()),
                export: false,
                kind: LayerKind::ConvBlock {
                    from: None,
                    out_c: 1024,
                    k: 3,
                    s: 2,
                },
            },
            LayerInit {
                name: None,
                export: false,
                kind: LayerKind::Spp {
                    from: None,
                    out_c: 1024,
                    ks: vec![5, 9, 13],
                },
            },
            // head p5
            LayerInit {
                name: Some("head-p5".into()),
                export: false,
                kind: LayerKind::BottleneckCsp {
                    from: None,
                    repeat: 3,
                    shortcut: false,
                },
            },
            // head p4
            LayerInit {
                name: None,
                export: false,
                kind: LayerKind::ConvBlock {
                    from: None,
                    out_c: 512,
                    k: 1,
                    s: 1,
                },
            },
            LayerInit {
                name: Some("upsample-p4".into()),
                export: false,
                kind: LayerKind::Upsample {
                    from: None,
                    scale_factor: 2.0,
                },
            },
            LayerInit {
                name: None,
                export: false,
                kind: LayerKind::Concat {
                    from: vec!["backbone-p4".into(), "upsample-p4".into()],
                },
            },
            LayerInit {
                name: Some("head-p4".into()),
                export: false,
                kind: LayerKind::BottleneckCsp {
                    from: None,
                    repeat: 3,
                    shortcut: false,
                },
            },
            // head p3
            LayerInit {
                name: None,
                export: false,
                kind: LayerKind::ConvBlock {
                    from: None,
                    out_c: 256,
                    k: 1,
                    s: 1,
                },
            },
            LayerInit {
                name: Some("upsample-p3".into()),
                export: false,
                kind: LayerKind::Upsample {
                    from: None,
                    scale_factor: 2.0,
                },
            },
            LayerInit {
                name: None,
                export: false,
                kind: LayerKind::Concat {
                    from: vec!["backbone-p3".into(), "upsample-p3".into()],
                },
            },
            LayerInit {
                name: None,
                export: false,
                kind: LayerKind::BottleneckCsp {
                    from: None,
                    repeat: 3,
                    shortcut: false,
                },
            },
            LayerInit {
                name: None,
                export: true,
                kind: LayerKind::HeadConv2d {
                    from: None,
                    k: 1,
                    s: 1,
                },
            },
            // head p2
            LayerInit {
                name: Some("head-conv-p2".into()),
                export: false,
                kind: LayerKind::ConvBlock {
                    from: None,
                    out_c: 256,
                    k: 3,
                    s: 2,
                },
            },
            LayerInit {
                name: None,
                export: false,
                kind: LayerKind::Concat {
                    from: vec!["head-conv-p2".into(), "head-p4".into()],
                },
            },
            LayerInit {
                name: None,
                export: false,
                kind: LayerKind::BottleneckCsp {
                    from: None,
                    repeat: 3,
                    shortcut: false,
                },
            },
            LayerInit {
                name: None,
                export: true,
                kind: LayerKind::HeadConv2d {
                    from: None,
                    k: 1,
                    s: 1,
                },
            },
            // head p1
            LayerInit {
                name: Some("head-conv-p1".into()),
                export: false,
                kind: LayerKind::ConvBlock {
                    from: None,
                    out_c: 512,
                    k: 3,
                    s: 2,
                },
            },
            LayerInit {
                name: None,
                export: false,
                kind: LayerKind::Concat {
                    from: vec!["head-conv-p1".into(), "head-p5".into()],
                },
            },
            LayerInit {
                name: None,
                export: false,
                kind: LayerKind::BottleneckCsp {
                    from: None,
                    repeat: 3,
                    shortcut: false,
                },
            },
            LayerInit {
                name: None,
                export: true,
                kind: LayerKind::HeadConv2d {
                    from: None,
                    k: 1,
                    s: 1,
                },
            },
        ],
        anchors: vec![
            vec![(116, 90), (156, 198), (373, 326)], // P5/32
            vec![(30, 61), (62, 45), (59, 119)],     // P4/1/6
            vec![(10, 13), (16, 30), (33, 23)],      // P3/8
        ],
    }
}

pub fn yolo_v5<'p, P>(path: P)
where
    P: Borrow<nn::Path<'p>>,
{
    let init = yolo_v5_init();
    let model = init.build(path);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YoloInit {
    pub input_channels: usize,
    pub num_classes: usize,
    pub depth_multiple: f64,
    pub width_multiple: f64,
    pub layers: Vec<LayerInit>,
    pub anchors: Vec<Vec<(usize, usize)>>,
}

impl YoloInit {
    pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor) -> Tensor>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();
        let Self {
            input_channels,
            num_classes,
            depth_multiple,
            width_multiple,
            layers,
            anchors,
        } = self;

        assert!(input_channels > 0);
        assert!(num_classes > 0);
        assert!(depth_multiple.is_finite() && depth_multiple > 0.0);
        assert!(width_multiple.is_finite() && width_multiple > 0.0);
        let num_anchors = anchors.len();
        let num_outputs_per_anchor = num_classes + 5;

        let scale_channel = |channel: usize| -> usize {
            let divisor = 8;
            ((channel as f64 * width_multiple / divisor as f64).ceil() * divisor as f64) as usize
        };

        // compute layer name to index correspondence
        // name -> layer_index
        let layer_names = layers
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| {
                let layer_index = index + 1;
                layer.name.as_ref().map(|name| (name.as_str(), layer_index))
            })
            .collect::<HashMap<&str, usize>>();

        // compute input indexes per layer
        // layer_index -> (from_index?, from_indexes?)
        let input_indexes = layers
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let layer_index = index + 1;
                let kind = &layer.kind;

                let (from_index_opt, from_indexes_opt) =
                    match (kind.from_name(), kind.from_multiple_names()) {
                        (Some(name), None) => {
                            let from_index = *layer_names
                                .get(name)
                                .expect(&format!(r#"undefined layer name "{}""#, name));
                            (Some(from_index), None)
                        }
                        (None, None) => {
                            let from_index = layer_index - 1;
                            (Some(from_index), None)
                        }
                        (None, Some(names)) => {
                            // let _: HashMap<&str, usize> = layer_names;
                            let from_indexes = names
                                .iter()
                                .map(|name| {
                                    *layer_names
                                        .get(name.as_str())
                                        .expect(&format!(r#"undefined layer name "{}""#, name))
                                })
                                .collect::<Vec<_>>();
                            (None, Some(from_indexes))
                        }
                        _ => unreachable!("please report bug"),
                    };

                (layer_index, (from_index_opt, from_indexes_opt))
            })
            .collect::<HashMap<usize, (Option<usize>, Option<Vec<usize>>)>>();

        // compute output channels per layer
        // layer_index -> (in_c?, out_c)
        let in_out_channels: HashMap<usize, (Option<usize>, usize)> =
            layers.iter().enumerate().fold(
                iter::once((0, (None, input_channels))).collect::<HashMap<_, _>>(),
                |mut channels, (index, layer)| {
                    let layer_index = index + 1;
                    let (from_index_opt, from_indexes_opt) = &input_indexes[&layer_index];

                    match layer.kind {
                        LayerKind::Focus { out_c, .. } => {
                            let from_index = from_index_opt.unwrap();
                            let in_c = channels[&from_index].1;
                            let out_c = scale_channel(out_c);
                            channels.insert(layer_index, (Some(in_c), out_c));
                        }
                        LayerKind::ConvBlock { out_c, .. } => {
                            let from_index = from_index_opt.unwrap();
                            let in_c = channels[&from_index].1;
                            let out_c = scale_channel(out_c);
                            channels.insert(layer_index, (Some(in_c), out_c));
                        }
                        LayerKind::Bottleneck { .. } => {
                            let from_index = from_index_opt.unwrap();
                            let in_c = channels[&from_index].1;
                            let out_c = in_c;
                            channels.insert(layer_index, (Some(in_c), out_c));
                        }
                        LayerKind::BottleneckCsp { .. } => {
                            let from_index = from_index_opt.unwrap();
                            let in_c = channels[&from_index].1;
                            let out_c = in_c;
                            channels.insert(layer_index, (Some(in_c), out_c));
                        }
                        LayerKind::Spp { out_c, .. } => {
                            let from_index = from_index_opt.unwrap();
                            let in_c = channels[&from_index].1;
                            let out_c = scale_channel(out_c);
                            channels.insert(layer_index, (Some(in_c), out_c));
                        }
                        LayerKind::HeadConv2d { .. } => {
                            let from_index = from_index_opt.unwrap();
                            let in_c = channels[&from_index].1;
                            let out_c = num_anchors * num_outputs_per_anchor;
                            channels.insert(layer_index, (Some(in_c), out_c));
                        }
                        LayerKind::Upsample { .. } => {
                            let from_index = from_index_opt.unwrap();
                            let in_c = channels[&from_index].1;
                            let out_c = in_c;
                            channels.insert(layer_index, (Some(in_c), out_c));
                        }
                        LayerKind::Concat { .. } => {
                            let from_indexes = from_indexes_opt.as_ref().unwrap();
                            let out_c = from_indexes
                                .iter()
                                .cloned()
                                .map(|index| channels[&index].1)
                                .sum();
                            channels.insert(layer_index, (None, out_c));
                        }
                    }

                    channels
                },
            );

        let export_indexes: Vec<usize> = layers.iter()
            .enumerate()
            .filter(|(index, layer)| layer.export)
            .map(|(index, _layer)| {
                let layer_index = index + 1;
                let out_c = in_out_channels[&layer_index].1;
                assert_eq!(out_c, num_anchors * num_outputs_per_anchor, "the exported layer must have exactly (n_anchros * (n_classes + 5)) output channels");
                layer_index
            }).collect::<Vec<_>>();

        // build modules for each layer
        // layer_index -> module
        let moduels = layers
            .iter()
            .enumerate()
            .map(|(index, layer_init)| {
                // locals
                let layer_index = index + 1;
                let LayerInit { kind, .. } = layer_init;

                let (from_index_opt, from_indexes_opt) = &input_indexes[&layer_index];
                let (in_c_opt, out_c): (Option<usize>, usize) = in_out_channels[&layer_index];

                // build layer
                let module = match *kind {
                    LayerKind::Focus { k, .. } => {
                        let from_index = from_index_opt.unwrap();
                        let in_c = in_c_opt.unwrap();
                        YoloModule::single(from_index, FocusInit { in_c, out_c, k }.build(path))
                    }
                    LayerKind::ConvBlock { k, s, .. } => {
                        let from_index = from_index_opt.unwrap();
                        let in_c = in_c_opt.unwrap();

                        YoloModule::single(
                            from_index,
                            ConvBlockInit {
                                k,
                                s,
                                ..ConvBlockInit::new(in_c, out_c)
                            }
                            .build(path),
                        )
                    }
                    LayerKind::Bottleneck { repeat, .. } => {
                        let from_index = from_index_opt.unwrap();
                        let in_c = in_c_opt.unwrap();
                        let repeat = ((repeat as f64 * depth_multiple).round() as usize).max(1);
                        let bottlenecks = (0..repeat)
                            .into_iter()
                            .map(|_| BottleneckInit::new(in_c, out_c).build(path))
                            .collect::<Vec<_>>();

                        YoloModule::single(from_index, move |xs, train| {
                            bottlenecks
                                .iter()
                                .fold(xs.shallow_clone(), |xs, block| block(&xs, train))
                        })
                    }
                    LayerKind::BottleneckCsp {
                        repeat, shortcut, ..
                    } => {
                        let from_index = from_index_opt.unwrap();
                        let in_c = in_c_opt.unwrap();

                        YoloModule::single(
                            from_index,
                            BottleneckCspInit {
                                repeat,
                                shortcut,
                                ..BottleneckCspInit::new(in_c, out_c)
                            }
                            .build(path),
                        )
                    }
                    LayerKind::Spp { ref ks, .. } => {
                        let from_index = from_index_opt.unwrap();
                        let in_c = in_c_opt.unwrap();

                        YoloModule::single(
                            from_index,
                            SppInit {
                                in_c,
                                out_c,
                                ks: ks.to_vec(),
                            }
                            .build(path),
                        )
                    }
                    LayerKind::HeadConv2d { k, s, .. } => {
                        let from_index = from_index_opt.unwrap();
                        let in_c = in_c_opt.unwrap();
                        let conv = nn::conv2d(
                            path,
                            in_c as i64,
                            out_c as i64,
                            k as i64,
                            nn::ConvConfig {
                                stride: s as i64,
                                ..Default::default()
                            },
                        );

                        YoloModule::single(from_index, move |xs, train| xs.apply_t(&conv, train))
                    }
                    LayerKind::Upsample { scale_factor, .. } => {
                        let from_index = from_index_opt.unwrap();

                        YoloModule::single(from_index, move |xs, _train| {
                            xs.upsample_nearest2d(
                                &[], // TODO
                                scale_factor,
                                scale_factor,
                            )
                        })
                    }
                    LayerKind::Concat { .. } => {
                        let from_indexes = from_indexes_opt.as_ref().unwrap();
                        YoloModule::multi(from_indexes.to_vec(), move |tensors, _train| {
                            Tensor::cat(tensors, 1)
                        })
                    }
                };

                (layer_index, module)
            })
            .collect::<HashMap<usize, YoloModule>>();

        todo!();
    }
}

pub enum YoloModule {
    Single(usize, Box<dyn 'static + Fn(&Tensor, bool) -> Tensor>),
    Multi(
        Vec<usize>,
        Box<dyn 'static + Fn(&[&Tensor], bool) -> Tensor>,
    ),
}

impl YoloModule {
    pub fn single<F>(from_index: usize, f: F) -> Self
    where
        F: 'static + Fn(&Tensor, bool) -> Tensor,
    {
        Self::Single(from_index, Box::new(f))
    }

    pub fn multi<F>(from_indexes: Vec<usize>, f: F) -> Self
    where
        F: 'static + Fn(&[&Tensor], bool) -> Tensor,
    {
        Self::Multi(from_indexes, Box::new(f))
    }
}

mod layers {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LayerInit {
        pub name: Option<String>,
        pub export: bool,
        pub kind: LayerKind,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
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
        },
        Upsample {
            from: Option<String>,
            scale_factor: f64,
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

    #[derive(Debug, Clone)]
    pub struct ConvBlockInit {
        pub in_c: usize,
        pub out_c: usize,
        pub k: usize,
        pub s: usize,
        pub g: usize,
        pub with_activation: bool,
    }

    impl ConvBlockInit {
        pub fn new(in_c: usize, out_c: usize) -> Self {
            Self {
                in_c,
                out_c,
                k: 1,
                s: 1,
                g: 1,
                with_activation: true,
            }
        }

        pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor, bool) -> Tensor>
        where
            P: Borrow<nn::Path<'p>>,
        {
            let path = path.borrow();

            let Self {
                in_c,
                out_c,
                k,
                s,
                g,
                with_activation,
            } = self;

            let conv = nn::conv2d(
                path,
                in_c as i64,
                out_c as i64,
                k as i64,
                nn::ConvConfig {
                    stride: s as i64,
                    padding: k as i64 / 2,
                    groups: g as i64,
                    bias: false,
                    ..Default::default()
                },
            );
            let bn = nn::batch_norm2d(path, out_c as i64, Default::default());

            Box::new(move |xs, train| {
                let xs = xs.apply(&conv).apply_t(&bn, train);
                if with_activation {
                    xs.leaky_relu()
                } else {
                    xs
                }
            })
        }
    }

    #[derive(Debug, Clone)]
    pub struct BottleneckInit {
        pub in_c: usize,
        pub out_c: usize,
        pub shortcut: bool,
        pub g: usize,
        pub expansion: R64,
    }

    impl BottleneckInit {
        pub fn new(in_c: usize, out_c: usize) -> Self {
            Self {
                in_c,
                out_c,
                shortcut: true,
                g: 1,
                expansion: R64::new(0.5),
            }
        }

        pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor, bool) -> Tensor>
        where
            P: Borrow<nn::Path<'p>>,
        {
            let path = path.borrow();

            let Self {
                in_c,
                out_c,
                shortcut,
                g,
                expansion,
            } = self;

            let intermediate_channels = (out_c as f64 * expansion.raw()) as usize;

            let conv1 = ConvBlockInit {
                k: 1,
                s: 1,
                ..ConvBlockInit::new(in_c, intermediate_channels)
            }
            .build(path);
            let conv2 = ConvBlockInit {
                k: 3,
                s: 1,
                ..ConvBlockInit::new(intermediate_channels, out_c)
            }
            .build(path);
            let with_add = shortcut && in_c == out_c;

            Box::new(move |xs, train| {
                let ys = conv1(xs, train);
                let ys = conv2(&xs, train);
                if with_add {
                    xs + &ys
                } else {
                    ys
                }
            })
        }
    }

    #[derive(Debug, Clone)]
    pub struct BottleneckCspInit {
        pub in_c: usize,
        pub out_c: usize,
        pub repeat: usize,
        pub shortcut: bool,
        pub g: usize,
        pub expansion: R64,
    }

    impl BottleneckCspInit {
        pub fn new(in_c: usize, out_c: usize) -> Self {
            Self {
                in_c,
                out_c,
                repeat: 1,
                shortcut: true,
                g: 1,
                expansion: R64::new(0.5),
            }
        }

        pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor, bool) -> Tensor>
        where
            P: Borrow<nn::Path<'p>>,
        {
            let path = path.borrow();

            let Self {
                in_c,
                out_c,
                repeat,
                shortcut,
                g,
                expansion,
            } = self;
            debug_assert!(repeat > 0);

            let intermediate_channels = (out_c as f64 * expansion.raw()) as usize;

            let conv1 = ConvBlockInit {
                k: 1,
                s: 1,
                ..ConvBlockInit::new(in_c, intermediate_channels)
            }
            .build(path);
            let conv2 = nn::conv2d(
                path,
                in_c as i64,
                intermediate_channels as i64,
                1,
                nn::ConvConfig {
                    stride: 1,
                    bias: false,
                    ..Default::default()
                },
            );
            let conv3 = nn::conv2d(
                path,
                intermediate_channels as i64,
                intermediate_channels as i64,
                1,
                nn::ConvConfig {
                    stride: 1,
                    bias: false,
                    ..Default::default()
                },
            );
            let conv4 = ConvBlockInit {
                k: 1,
                s: 1,
                ..ConvBlockInit::new(out_c, out_c)
            }
            .build(path);
            let bn = nn::batch_norm2d(path, intermediate_channels as i64 * 2, Default::default());
            let bottlenecks = (0..repeat)
                .map(|_| {
                    BottleneckInit {
                        shortcut,
                        g,
                        expansion: R64::new(1.0),
                        ..BottleneckInit::new(intermediate_channels, intermediate_channels)
                    }
                    .build(path)
                })
                .collect::<Vec<_>>();

            Box::new(move |xs, train| {
                let y1 = {
                    let y = conv1(xs, train);
                    let y = bottlenecks
                        .iter()
                        .fold(y, |input, block| block(&input, train));
                    y.apply_t(&conv3, train)
                };
                let y2 = xs.apply_t(&conv2, train);
                conv4(
                    &Tensor::cat(&[y1, y2], 1).apply_t(&bn, train).leaky_relu(),
                    train,
                )
            })
        }
    }

    pub struct SppInit {
        pub in_c: usize,
        pub out_c: usize,
        pub ks: Vec<usize>,
    }

    impl SppInit {
        pub fn new(in_c: usize, out_c: usize) -> Self {
            Self {
                in_c,
                out_c,
                ks: vec![5, 9, 13],
            }
        }

        pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor, bool) -> Tensor>
        where
            P: Borrow<nn::Path<'p>>,
        {
            let path = path.borrow();

            let Self { in_c, out_c, ks } = self;
            let intermediate_channels = in_c / 2;

            let conv1 = ConvBlockInit {
                k: 1,
                s: 1,
                ..ConvBlockInit::new(in_c, intermediate_channels)
            }
            .build(path);

            let conv2 = ConvBlockInit {
                k: 1,
                s: 1,
                ..ConvBlockInit::new(intermediate_channels * ks.len(), out_c)
            }
            .build(path);

            Box::new(move |xs, train| {
                let transformed_xs = conv1(xs, train);

                let first_iter = iter::once(xs.shallow_clone());
                let pyramid_iter = ks.iter().cloned().map(|k| {
                    let k = k as i64;
                    let padding = k / 2;
                    let s = 1;
                    let dilation = 1;
                    let ceil_mode = false;
                    transformed_xs.max_pool2d(
                        &[k, k],
                        &[s, s],
                        &[padding, padding],
                        &[dilation, dilation],
                        ceil_mode,
                    )
                });
                let cat_xs = Tensor::cat(&first_iter.chain(pyramid_iter).collect::<Vec<_>>(), 1);
                conv2(&cat_xs, train)
            })
        }
    }

    #[derive(Debug, Clone)]
    pub struct FocusInit {
        pub in_c: usize,
        pub out_c: usize,
        pub k: usize,
    }

    impl FocusInit {
        pub fn new(in_c: usize, out_c: usize) -> Self {
            Self { in_c, out_c, k: 1 }
        }

        pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor, bool) -> Tensor>
        where
            P: Borrow<nn::Path<'p>>,
        {
            let path = path.borrow();

            let Self { in_c, out_c, k } = self;

            let conv = ConvBlockInit {
                k,
                s: 1,
                ..ConvBlockInit::new(in_c * 4, out_c)
            }
            .build(path);

            Box::new(move |xs, train| {
                let xs = Tensor::cat(
                    &[
                        xs.slice(2, 0, -1, 2).slice(3, 0, -1, 2),
                        xs.slice(2, 1, -1, 2).slice(3, 0, -1, 2),
                        xs.slice(2, 0, -1, 2).slice(3, 1, -1, 2),
                    ],
                    1,
                );
                conv(&xs, train)
            })
        }
    }

    #[derive(Debug, Clone)]
    struct DetectInit {
        pub num_classes: usize,
        pub anchors: Vec<Vec<(usize, usize)>>,
    }

    impl DetectInit {
        pub fn build<'p, P>(
            self,
            path: P,
        ) -> Box<dyn FnMut(&[&Tensor], bool, i64, i64) -> (Tensor, Option<Tensor>)>
        where
            P: Borrow<nn::Path<'p>>,
        {
            let path = path.borrow();
            let device = path.device();

            let Self {
                num_classes,
                anchors,
            } = self;

            assert!(anchors
                .iter()
                .all(|anchor| anchor.len() == anchors[0].len()));
            let num_anchors = anchors[0].len() as i64;
            let num_outputs_per_anchor = num_classes as i64 + 5;
            let num_detections = anchors.len() as i64;

            let anchors_tensor = Tensor::of_slice(
                &anchors
                    .iter()
                    .flat_map(|anchor| {
                        anchor
                            .iter()
                            .cloned()
                            .flat_map(|(y, x)| vec![y as i64, x as i64])
                    })
                    .collect::<Vec<_>>(),
            );
            let anchors_grid_tensor =
                anchors_tensor.view(&[num_detections, 1, num_anchors, 1, 1, 2] as &[_]);
            let mut grid_opts = (0..num_detections).map(|_| None).collect::<Vec<_>>();

            let make_grid = |height: i64, width: i64, device: Device| -> Tensor {
                let grids = Tensor::meshgrid(&[
                    Tensor::arange(height, (Kind::Float, device)),
                    Tensor::arange(width, (Kind::Float, device)),
                ]);
                Tensor::stack(&[&grids[0], &grids[1]], 2).view(&[1, 1, height, width, 2] as &[_])
            };

            Box::new(
                move |tensors: &[&Tensor], train: bool, image_height: i64, image_width: i64| {
                    debug_assert_eq!(tensors.len() as i64, num_detections);

                    let training_outputs = tensors
                        .iter()
                        .cloned()
                        .map(|xs| {
                            let (batch_size, channels, height, width) = match xs.size().as_slice() {
                                &[bs, c, h, w] => (bs, c, h, w),
                                _ => unreachable!(),
                            };
                            debug_assert_eq!(channels, num_anchors * num_outputs_per_anchor);

                            // to shape [bsize, n_anchors, height, width, n_outputs]
                            let training_outputs = xs
                                .view(&[
                                    batch_size,
                                    num_anchors,
                                    num_outputs_per_anchor,
                                    height,
                                    width,
                                ] as &[_])
                                .permute(&[0, 1, 3, 4, 2]);

                            training_outputs
                        })
                        .collect::<Vec<_>>();

                    let inference_outputs_opt = if !train {
                        let strides = tensors
                            .iter()
                            .cloned()
                            .map(|xs| {
                                let (height, width) = match xs.size().as_slice() {
                                    &[_bs, _c, h, w] => (h, w),
                                    _ => unreachable!(),
                                };
                                (image_height / height, image_width / width)
                            })
                            .collect::<Vec<_>>();

                        let outputs = training_outputs
                            .iter()
                            .zip(strides.iter().cloned())
                            .enumerate()
                            .map(|(index, (xs, (height_stride, width_stride)))| {
                                let (batch_size, height, width) = match xs.size().as_slice() {
                                    &[bs, _n_a, h, w, _n_o] => (bs, h, w),
                                    _ => unreachable!(),
                                };

                                // prepare grid
                                let grid = {
                                    let grid_opt = &mut grid_opts[index];
                                    let grid = grid_opt
                                        .get_or_insert_with(|| make_grid(height, width, device));
                                    let grid = match grid.size().as_slice() {
                                        &[_, _, grid_height, grid_width, _] => {
                                            if grid_height != height || grid_width != width {
                                                grid_opt.replace(make_grid(height, width, device));
                                                grid_opt.as_mut().unwrap()
                                            } else {
                                                grid
                                            }
                                        }
                                        _ => unreachable!(),
                                    };
                                    grid
                                };

                                let stride_multiplier =
                                    Tensor::of_slice(&[height_stride, width_stride] as &[_])
                                        .view(&[1i64, 1, 1, 2] as &[_])
                                        .expand_as(grid);
                                let sigmoid = xs.sigmoid();
                                let position = sigmoid.i((.., .., .., .., 0..2)) * 2.0 - 0.5
                                    + &*grid * &stride_multiplier; // TODO: stride
                                let size = sigmoid.i((.., .., .., .., 2..4)).pow(2.0)
                                    * anchors_grid_tensor.select(0, index as i64);
                                let objectness = sigmoid.i((.., .., .., .., 4..5));
                                let classification = sigmoid.i((.., .., .., .., 5..));
                                let output =
                                    Tensor::stack(&[position, size, objectness, classification], 4)
                                        .view(&[batch_size, -1, num_outputs_per_anchor] as &[_]);

                                output
                            })
                            .collect::<Vec<_>>();

                        Some(outputs)
                    } else {
                        None
                    };

                    (
                        Tensor::stack(&training_outputs, 1),
                        inference_outputs_opt.map(|outputs| Tensor::stack(&outputs, 1)),
                    )
                },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yolov5_init_test() {
        let device = Device::cuda_if_available();
        let vs = nn::VarStore::new(device);
        let root = vs.root();
        yolo_v5(&root);
    }
}
