use super::*;
use crate::common::*;

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
    pub fn build<'p, P>(self, path: P) -> Result<Box<dyn FnMut(&Tensor, bool) -> YoloOutput + Send>>
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

        ensure!(input_channels > 0, "input_channels must be positive");
        ensure!(num_classes > 0, "num_classes must be positive");
        ensure!(depth_multiple > 0.0, "depth_multiple must be positive");
        ensure!(width_multiple > 0.0, "width_multiple must be positive");
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
        let func = Box::new(move |xs: &Tensor, train: bool| -> YoloOutput {
            let (_batch_size, _channels, height, width) = xs.size4().unwrap();
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
        });

        Ok(func)
    }
}

#[derive(Debug, TensorLike)]
pub struct YoloOutput {
    pub(crate) image_height: i64,
    pub(crate) image_width: i64,
    pub(crate) batch_size: i64,
    #[tensor_like(copy)]
    pub(crate) device: Device,
    /// Feature map sizes in grid units per layer.
    pub(crate) feature_sizes: Vec<(i64, i64)>,
    /// Grid sizes in pixels per layer.
    pub(crate) grid_sizes: Vec<(i64, i64)>,
    /// Detections indexed by (layer_index, anchor_index, col, row) measured in grid units
    pub(crate) detections: Vec<Detection>,
    pub(crate) anchors: Vec<Vec<(i64, i64)>>,
}

impl YoloOutput {
    pub fn image_height(&self) -> i64 {
        self.image_height
    }

    pub fn image_width(&self) -> i64 {
        self.image_width
    }

    pub fn detections(&self) -> &[Detection] {
        self.detections.as_slice()
    }

    pub fn detections_in_pixels(&self) -> Vec<Detection> {
        self.detections
            .shallow_clone()
            .into_iter()
            .map(|detection| {
                let Detection {
                    index,
                    position: position_grids,
                    size: size_grids,
                    objectness,
                    classification,
                } = detection;

                let layer_index = index.layer_index;
                let (grid_height, grid_width) = self.grid_sizes[layer_index];

                let grid_size_multiplier =
                    Tensor::of_slice(&[grid_height as i64, grid_width as i64]).view([1, 2]);
                let position_pixels = position_grids * &grid_size_multiplier;
                let size_pixels = size_grids * &grid_size_multiplier;

                let new_detection = Detection {
                    index,
                    position: position_pixels,
                    size: size_pixels,
                    objectness,
                    classification,
                };

                new_detection
            })
            .collect()
    }

    pub fn anchors(&self) -> &[Vec<(i64, i64)>] {
        self.anchors.as_slice()
    }

    pub fn batch_size(&self) -> i64 {
        self.batch_size
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, TensorLike)]
pub struct DetectionIndex {
    pub layer_index: usize,
    pub anchor_index: usize,
    pub grid_row: i64,
    pub grid_col: i64,
}

#[derive(Debug, TensorLike)]
pub struct Detection {
    pub index: DetectionIndex,
    pub position: Tensor,
    pub size: Tensor,
    pub objectness: Tensor,
    pub classification: Tensor,
}
