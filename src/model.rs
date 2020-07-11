use crate::common::*;
use layers::*;

#[derive(Debug, TensorLike)]
pub struct YoloOutput {
    image_height: i64,
    image_width: i64,
    #[tensor_like(copy)]
    device: Device,
    anchor_size_multipliers: Vec<Tensor>,
    feature_maps: Vec<Tensor>,
}

impl YoloOutput {
    pub fn image_height(&self) -> i64 {
        self.image_height
    }

    pub fn image_width(&self) -> i64 {
        self.image_width
    }

    pub fn feature_maps(&self) -> &[Tensor] {
        self.feature_maps.as_slice()
    }

    pub fn to_detections(&self) -> Vec<Detection> {
        let detections = self
            .feature_maps
            .iter()
            .enumerate()
            .map(|(index, xs)| {
                let (batch_size, height, width) = match xs.size().as_slice() {
                    &[b, _na, h, w, _no] => (b, h, w),
                    _ => unreachable!(),
                };

                // compute gride size
                let grid_height = self.image_height / height;
                let grid_width = self.image_width / width;

                // prepare grid
                let grid = {
                    let grids = Tensor::meshgrid(&[
                        Tensor::arange(height, (Kind::Float, self.device)),
                        Tensor::arange(width, (Kind::Float, self.device)),
                    ]);
                    Tensor::stack(&[&grids[0], &grids[1]], 2)
                        .view(&[1, 1, height, width, 2] as &[_])
                };

                let stride_multiplier = Tensor::of_slice(&[grid_height, grid_width] as &[_])
                    .view([1, 1, 1, 2])
                    .expand_as(&grid)
                    .to_device(self.device);

                // transform outputs
                let sigmoid = xs.sigmoid();
                let position =
                    (sigmoid.i((.., .., .., .., 0..2)) * 2.0 - 0.5 + &grid) * &stride_multiplier;
                let size = sigmoid.i((.., .., .., .., 2..4)).pow(2.0)
                    * &self.anchor_size_multipliers[index];
                let objectness = sigmoid.i((.., .., .., .., 4..5));
                let classification = sigmoid.i((.., .., .., .., 5..));

                let detection = Detection {
                    position,
                    size,
                    objectness,
                    classification,
                };

                detection
            })
            .collect::<Vec<_>>();

        detections
    }
}

#[derive(Debug, TensorLike)]
pub struct Detection {
    pub position: Tensor,
    pub size: Tensor,
    pub objectness: Tensor,
    pub classification: Tensor,
}

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
                    scale_factor: R64::new(2.0),
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
                    scale_factor: R64::new(2.0),
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

pub fn yolo_v5_small<'p, P>(
    path: P,
    input_channels: usize,
    num_classes: usize,
) -> Box<dyn FnMut(&Tensor, bool) -> YoloOutput>
where
    P: Borrow<nn::Path<'p>>,
{
    let init = yolo_v5_small_init(input_channels, num_classes);
    let model = init.build(path);
    model
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct YoloInit {
    pub input_channels: usize,
    pub num_classes: usize,
    pub depth_multiple: R64,
    pub width_multiple: R64,
    pub anchors: Vec<Vec<(usize, usize)>>,
    pub layers: Vec<LayerInit>,
}

impl YoloInit {
    pub fn build<'p, P>(self, path: P) -> Box<dyn FnMut(&Tensor, bool) -> YoloOutput>
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
        let depth_multiple = depth_multiple.raw();
        let width_multiple = width_multiple.raw();

        assert!(input_channels > 0);
        assert!(num_classes > 0);
        assert!(depth_multiple > 0.0);
        assert!(width_multiple > 0.0);
        let num_anchors = anchors.len();
        let num_outputs_per_anchor = num_classes + 5;

        let scale_channel = |channel: usize| -> usize {
            let divisor = 8;
            ((channel as f64 * width_multiple / divisor as f64).ceil() * divisor as f64) as usize
        };

        // annotate each layer with layer index
        // layer_index -> layer_config
        let layers = layers
            .into_iter()
            .enumerate()
            .map(|(index, layer)| {
                let layer_index = index + 1;
                (layer_index, layer)
            })
            .collect::<HashMap<usize, _>>();

        // compute layer name to index correspondence
        // name -> layer_index
        let layer_names = layers
            .iter()
            .filter_map(|(layer_index, layer)| {
                layer
                    .name
                    .as_ref()
                    .map(|name| (name.as_str(), *layer_index))
            })
            .collect::<HashMap<&str, usize>>();

        // compute input indexes per layer
        // layer_index -> (from_index?, from_indexes?)
        let input_indexes = layers
            .iter()
            .map(|(layer_index, layer)| {
                let kind = &layer.kind;

                let from_indexes = match (kind.from_name(), kind.from_multiple_names()) {
                    (Some(name), None) => {
                        let from_index = *layer_names
                            .get(name)
                            .expect(&format!(r#"undefined layer name "{}""#, name));
                        vec![from_index]
                    }
                    (None, None) => {
                        let from_index = layer_index - 1;
                        vec![from_index]
                    }
                    (None, Some(names)) => names
                        .iter()
                        .map(|name| {
                            *layer_names
                                .get(name.as_str())
                                .expect(&format!(r#"undefined layer name "{}""#, name))
                        })
                        .collect::<Vec<_>>(),
                    _ => unreachable!("please report bug"),
                };

                (*layer_index, from_indexes)
            })
            .chain(iter::once((0, vec![])))
            .collect::<HashMap<usize, Vec<usize>>>();

        // topological sort on layers
        let ordered_layer_indexes = {
            // type defs
            #[derive(Debug, Clone, PartialEq, Eq, Hash)]
            enum Mark {
                Temporary,
                Permanent,
            }

            #[derive(Debug, Clone)]
            struct State {
                marks: HashMap<usize, Mark>,
                topo_indexes: HashMap<usize, usize>,
            }

            // list directed edges among nodes
            // layer_index -> [output_indexes]
            let output_indexes = input_indexes
                .iter()
                .flat_map(|(layer_index, from_indexes)| {
                    from_indexes
                        .iter()
                        .cloned()
                        .map(move |from_index| (from_index, *layer_index))
                })
                .into_group_map();

            // recursive visiting funciton
            fn visit_fn(
                visit_index: usize,
                layer_index: usize,
                mut state: State,
                input_indexes: &HashMap<usize, Vec<usize>>,
                output_indexes: &HashMap<usize, Vec<usize>>,
            ) -> (usize, State) {
                debug_assert_eq!(state.marks.get(&layer_index), None);

                // if any one of incoming node is not visited, give up and visit later
                for from_layer_index in input_indexes[&layer_index].iter().cloned() {
                    if let None = state.marks.get(&from_layer_index) {
                        return (visit_index, state);
                    }
                }

                state.marks.insert(layer_index, Mark::Temporary);
                state.topo_indexes.insert(visit_index, layer_index);

                let mut next_index = visit_index + 1;
                let to_layer_indexes = match output_indexes.get(&layer_index) {
                    Some(indexes) => indexes,
                    None => return (next_index, state),
                };

                for to_layer_index in to_layer_indexes.iter().cloned() {
                    match state.marks.get(&to_layer_index) {
                        Some(Mark::Permanent) => {
                            continue;
                        }
                        Some(Mark::Temporary) => {
                            panic!("the model must not contain cycles");
                        }
                        None => {
                            let (new_next_index, new_state) = visit_fn(
                                next_index,
                                to_layer_index,
                                state,
                                input_indexes,
                                output_indexes,
                            );
                            next_index = new_next_index;
                            state = new_state;
                        }
                    }
                }

                state.marks.insert(layer_index, Mark::Permanent);
                (next_index, state)
            };

            let mut state = State {
                marks: HashMap::<usize, Mark>::new(),
                topo_indexes: HashMap::<usize, usize>::new(),
            };
            let (end_visit_index, new_state) =
                visit_fn(0, 0, state, &input_indexes, &output_indexes);
            state = new_state;

            for layer_index in (1..=layers.len()).rev() {
                if let None = state.marks.get(&layer_index) {
                    panic!("the model graph is not connected");
                }
            }

            (0..end_visit_index)
                .map(|topo_index| state.topo_indexes[&topo_index])
                .filter(|layer_index| *layer_index != 0) // exclude input layer
                .collect::<Vec<usize>>()
        };

        // compute output channels per layer
        // layer_index -> (in_c?, out_c)
        let in_out_channels: HashMap<usize, (Option<usize>, usize)> = ordered_layer_indexes
            .iter()
            .cloned()
            .map(|layer_index| (layer_index, &layers[&layer_index]))
            .fold(
                iter::once((0, (None, input_channels))).collect::<HashMap<_, _>>(),
                |mut channels, (layer_index, layer)| {
                    let from_indexes = &input_indexes[&layer_index];

                    match layer.kind {
                        LayerKind::Focus { out_c, .. } => {
                            debug_assert_eq!(from_indexes.len(), 1);
                            let from_index = from_indexes[0];
                            let in_c = channels[&from_index].1;
                            let out_c = scale_channel(out_c);
                            channels.insert(layer_index, (Some(in_c), out_c));
                        }
                        LayerKind::ConvBlock { out_c, .. } => {
                            debug_assert_eq!(from_indexes.len(), 1);
                            let from_index = from_indexes[0];
                            let in_c = channels[&from_index].1;
                            let out_c = scale_channel(out_c);
                            channels.insert(layer_index, (Some(in_c), out_c));
                        }
                        LayerKind::Bottleneck { .. } => {
                            debug_assert_eq!(from_indexes.len(), 1);
                            let from_index = from_indexes[0];
                            let in_c = channels[&from_index].1;
                            let out_c = in_c;
                            channels.insert(layer_index, (Some(in_c), out_c));
                        }
                        LayerKind::BottleneckCsp { .. } => {
                            debug_assert_eq!(from_indexes.len(), 1);
                            let from_index = from_indexes[0];
                            let in_c = channels[&from_index].1;
                            let out_c = in_c;
                            channels.insert(layer_index, (Some(in_c), out_c));
                        }
                        LayerKind::Spp { out_c, .. } => {
                            debug_assert_eq!(from_indexes.len(), 1);
                            let from_index = from_indexes[0];
                            let in_c = channels[&from_index].1;
                            let out_c = scale_channel(out_c);
                            channels.insert(layer_index, (Some(in_c), out_c));
                        }
                        LayerKind::HeadConv2d { .. } => {
                            debug_assert_eq!(from_indexes.len(), 1);
                            let from_index = from_indexes[0];
                            let in_c = channels[&from_index].1;
                            let out_c = num_anchors * num_outputs_per_anchor;
                            channels.insert(layer_index, (Some(in_c), out_c));
                        }
                        LayerKind::Upsample { .. } => {
                            debug_assert_eq!(from_indexes.len(), 1);
                            let from_index = from_indexes[0];
                            let in_c = channels[&from_index].1;
                            let out_c = in_c;
                            channels.insert(layer_index, (Some(in_c), out_c));
                        }
                        LayerKind::Concat { .. } => {
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

        // list of exported layer indexes
        let exported_indexes: Vec<usize> = layers.iter()
            .filter(|(_layer_index, layer)| layer.export)
            .map(|(layer_index, _layer)| {
                let out_c = in_out_channels[&layer_index].1;
                assert_eq!(out_c, num_anchors * num_outputs_per_anchor, "the exported layer must have exactly (n_anchros * (n_classes + 5)) output channels");
                *layer_index
            }).collect::<Vec<_>>();

        // build modules for each layer
        // layer_index -> module
        let modules = layers
            .iter()
            .map(|(layer_index, layer_init)| {
                // locals
                let layer_index = *layer_index;
                let LayerInit { kind, .. } = layer_init;

                let from_indexes = &input_indexes[&layer_index];
                let (in_c_opt, out_c): (Option<usize>, usize) = in_out_channels[&layer_index];

                // build layer
                let module = match *kind {
                    LayerKind::Focus { k, .. } => {
                        debug_assert_eq!(from_indexes.len(), 1);
                        let from_index = from_indexes[0];
                        let in_c = in_c_opt.unwrap();
                        YoloModule::single(from_index, FocusInit { in_c, out_c, k }.build(path))
                    }
                    LayerKind::ConvBlock { k, s, .. } => {
                        debug_assert_eq!(from_indexes.len(), 1);
                        let from_index = from_indexes[0];
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
                        debug_assert_eq!(from_indexes.len(), 1);
                        let from_index = from_indexes[0];
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
                        debug_assert_eq!(from_indexes.len(), 1);
                        let from_index = from_indexes[0];
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
                        debug_assert_eq!(from_indexes.len(), 1);
                        let from_index = from_indexes[0];
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
                        debug_assert_eq!(from_indexes.len(), 1);
                        let from_index = from_indexes[0];
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
                        let scale_factor = scale_factor.raw();
                        debug_assert_eq!(from_indexes.len(), 1);
                        let from_index = from_indexes[0];

                        YoloModule::single(from_index, move |xs, _train| {
                            let (height, width) = match xs.size().as_slice() {
                                &[_bsize, _channels, height, width] => (height, width),
                                _ => unreachable!(),
                            };

                            let new_height = (height as f64 * scale_factor) as i64;
                            let new_width = (width as f64 * scale_factor) as i64;

                            xs.upsample_nearest2d(
                                &[new_height, new_width],
                                Some(scale_factor),
                                Some(scale_factor),
                            )
                        })
                    }
                    LayerKind::Concat { .. } => {
                        YoloModule::multi(from_indexes.to_vec(), move |tensors, _train| {
                            Tensor::cat(tensors, 1)
                        })
                    }
                };

                (layer_index, module)
            })
            .collect::<HashMap<usize, YoloModule>>();

        // construct detection head
        let mut detection_module = DetectInit {
            num_classes,
            anchors,
        }
        .build(path);

        // construct module function
        Box::new(move |xs: &Tensor, train: bool| -> YoloOutput {
            let (height, width) = match xs.size().as_slice() {
                &[_bsize, _channels, height, width] => (height, width),
                _ => unreachable!(),
            };
            let mut tmp_tensors =
                iter::once((0, xs.shallow_clone())).collect::<HashMap<usize, Tensor>>();
            let mut exported_tensors = vec![];

            // run the network
            ordered_layer_indexes
                .iter()
                .cloned()
                .for_each(|layer_index| {
                    let module = &modules[&layer_index];
                    let inputs = input_indexes[&layer_index]
                        .iter()
                        .cloned()
                        .map(|from_index| &tmp_tensors[&from_index])
                        .collect::<Vec<_>>();

                    // println!(
                    //     "{}\t{:?}",
                    //     layer_index,
                    //     inputs
                    //         .iter()
                    //         .map(|tensor| tensor.size())
                    //         .collect::<Vec<_>>()
                    // );

                    let output = module.forward_t(inputs.as_slice(), train);
                    if exported_indexes.contains(&layer_index) {
                        exported_tensors.push(output.shallow_clone());
                    }
                    tmp_tensors.insert(layer_index, output);
                });

            // run detection module
            let exported_tensors = exported_tensors.iter().collect::<Vec<_>>();
            detection_module(exported_tensors.as_slice(), train, height, width)
        })
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

    pub fn forward_t<T>(&self, inputs: &[T], train: bool) -> Tensor
    where
        T: Borrow<Tensor>,
    {
        match self {
            Self::Single(_, module_fn) => {
                debug_assert_eq!(inputs.len(), 1);
                module_fn(inputs[0].borrow(), train)
            }
            Self::Multi(_, module_fn) => {
                let inputs = inputs
                    .iter()
                    .map(|tensor| tensor.borrow())
                    .collect::<Vec<_>>();
                module_fn(&inputs, train)
            }
        }
    }
}

mod layers {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct LayerInit {
        pub name: Option<String>,
        pub export: bool,
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
                g,
                ..ConvBlockInit::new(intermediate_channels, out_c)
            }
            .build(path);
            let with_add = shortcut && in_c == out_c;

            Box::new(move |xs, train| {
                let ys = conv1(xs, train);
                let ys = conv2(&ys, train);
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
                ..ConvBlockInit::new(intermediate_channels * (ks.len() + 1), out_c)
            }
            .build(path);

            Box::new(move |xs, train| {
                let transformed_xs = conv1(xs, train);

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
                let cat_xs = Tensor::cat(
                    &iter::once(transformed_xs.shallow_clone())
                        .chain(pyramid_iter)
                        .collect::<Vec<_>>(),
                    1,
                );

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
                let (height, width) = match xs.size().as_slice() {
                    &[_bsize, _channels, height, width] => (height, width),
                    _ => unreachable!(),
                };

                let xs = Tensor::cat(
                    &[
                        xs.slice(2, 0, height, 2).slice(3, 0, width, 2),
                        xs.slice(2, 1, height, 2).slice(3, 0, width, 2),
                        xs.slice(2, 0, height, 2).slice(3, 1, width, 2),
                        xs.slice(2, 1, height, 2).slice(3, 1, width, 2),
                    ],
                    1,
                );
                conv(&xs, train)
            })
        }
    }

    #[derive(Debug, Clone)]
    pub struct DetectInit {
        pub num_classes: usize,
        pub anchors: Vec<Vec<(usize, usize)>>,
    }

    impl DetectInit {
        pub fn build<'p, P>(
            self,
            path: P,
        ) -> Box<dyn FnMut(&[&Tensor], bool, i64, i64) -> YoloOutput>
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

            let anchor_size_multipliers = anchors
                .iter()
                .cloned()
                .map(|sizes| {
                    Tensor::of_slice(
                        &sizes
                            .iter()
                            .cloned()
                            .flat_map(|(y, x)| vec![y, x])
                            .map(|component| component as i64)
                            .collect::<Vec<_>>(),
                    )
                    .to_device(device)
                    .view([1, num_anchors, 1, 1, 2])
                })
                .collect::<Vec<_>>();

            Box::new(
                move |tensors: &[&Tensor], train: bool, image_height: i64, image_width: i64| {
                    debug_assert_eq!(tensors.len() as i64, num_detections);

                    let feature_maps = tensors
                        .iter()
                        .cloned()
                        .map(|xs| {
                            let (batch_size, channels, height, width) = xs.size4().unwrap();
                            debug_assert_eq!(channels, num_anchors * num_outputs_per_anchor);

                            let outputs = xs
                                .view(&[
                                    batch_size,
                                    num_anchors,
                                    num_outputs_per_anchor,
                                    height,
                                    width,
                                ] as &[_])
                                .permute(&[0, 1, 3, 4, 2]);

                            // output shape [bsize, n_anchors, height, width, n_outputs]
                            outputs
                        })
                        .collect::<Vec<_>>();

                    YoloOutput {
                        image_height,
                        image_width,
                        device,
                        anchor_size_multipliers: anchor_size_multipliers.shallow_clone(),
                        feature_maps,
                    }
                },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use failure::Fallible;

    #[test]
    fn yolo_v5_small_serde_test() -> Fallible<()> {
        let init = yolo_v5_small_init();
        let text = serde_json::to_string_pretty(&init)?;
        println!("{}", text);
        let recovered = serde_json::from_str(&text)?;
        assert_eq!(init, recovered);
        Ok(())
    }

    #[test]
    fn yolo_v5_small_test() {
        let device = Device::cuda_if_available();
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        let mut yolo_fn = yolo_v5_small(&root);

        for _ in 0..10 {
            let input = Tensor::randn(
                &[32, 3, 224, 224],
                (Kind::Float, Device::cuda_if_available()),
            );
            let instant = std::time::Instant::now();
            let (train_outputs, inference_outputs_opt) = yolo_fn(&input, false);

            let train_output_shapes = train_outputs
                .iter()
                .map(|tensor| tensor.size())
                .collect::<Vec<_>>();
            println!("train output shapes: {:?}", train_output_shapes);

            if let Some(output) = inference_outputs_opt {
                println!("inference output shape: {:?}", output.size());
                let expect = train_output_shapes
                    .iter()
                    .map(|shape| match shape.as_slice() {
                        &[_bsize, channels, height, width, _outputs] => channels * height * width,
                        _ => unreachable!(),
                    })
                    .sum::<i64>();

                match output.size().as_slice() {
                    &[_bsize, found, _] => assert_eq!(expect, found),
                    _ => unreachable!(),
                }
            }

            println!("elapsed {:?}", instant.elapsed());
        }
    }
}
