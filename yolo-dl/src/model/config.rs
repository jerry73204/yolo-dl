use super::*;
use crate::{common::*, utils::PixelSize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct YoloInit {
    pub input_channels: usize,
    pub num_classes: usize,
    pub depth_multiple: R64,
    pub width_multiple: R64,
    pub layers: Vec<LayerInit>,
}

impl YoloInit {
    pub fn build<'p, P>(self, path: P) -> Result<YoloModel>
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
        } = self;
        let depth_multiple = depth_multiple.raw();
        let width_multiple = width_multiple.raw();

        ensure!(input_channels > 0, "input_channels must be positive");
        ensure!(num_classes > 0, "num_classes must be positive");
        ensure!(depth_multiple > 0.0, "depth_multiple must be positive");
        ensure!(width_multiple > 0.0, "width_multiple must be positive");
        let num_outputs_per_anchor = num_classes + 5;

        let scale_channel = |channel: usize| -> usize {
            let divisor = 8;
            ((channel as f64 * width_multiple / divisor as f64).ceil() * divisor as f64) as usize
        };

        // annotate each layer with layer index
        // layer_index -> layer_config
        let index_to_config: IndexMap<usize, _> = layers
            .into_iter()
            .enumerate()
            .map(|(index, layer)| {
                let layer_index = index + 1;
                (layer_index, layer)
            })
            .collect();

        // compute layer name to index correspondence
        // name -> layer_index
        let name_to_index: IndexMap<&str, usize> = index_to_config
            .iter()
            .filter_map(|(layer_index, layer)| {
                layer
                    .name
                    .as_ref()
                    .map(|name| (name.as_str(), *layer_index))
            })
            .collect();

        // compute input indexes per layer
        // layer_index -> (from_indexes)
        let mut index_to_inputs: IndexMap<usize, Vec<usize>> = index_to_config
            .iter()
            .map(|(layer_index, layer)| {
                let kind = &layer.kind;

                let from_indexes = match (kind.from_name(), kind.from_multiple_names()) {
                    (Some(name), None) => {
                        let from_index = *name_to_index
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
                            *name_to_index
                                .get(name.as_str())
                                .expect(&format!(r#"undefined layer name "{}""#, name))
                        })
                        .collect_vec(),
                    _ => unreachable!("please report bug"),
                };

                (*layer_index, from_indexes)
            })
            .chain(iter::once((0, vec![])))
            .collect();

        // topological sort on layers
        // ordered list of layer indexes
        let ordered_layer_indexes: Vec<usize> = {
            // type defs
            #[derive(Debug, Clone, PartialEq, Eq, Hash)]
            enum Mark {
                Temporary,
                Permanent,
            }

            #[derive(Debug, Clone)]
            struct State {
                marks: IndexMap<usize, Mark>,
                topo_indexes: IndexMap<usize, usize>,
            }

            // list directed edges among nodes
            // layer_index -> [output_indexes]
            let output_indexes: IndexMap<_, _> = index_to_inputs
                .iter()
                .flat_map(|(layer_index, from_indexes)| {
                    from_indexes
                        .iter()
                        .cloned()
                        .map(move |from_index| (from_index, *layer_index))
                })
                .into_group_map()
                .into_iter()
                .map(|(from_index, mut layer_indexes)| {
                    layer_indexes.sort_by_cached_key(|index| *index);
                    (from_index, layer_indexes)
                })
                .collect();

            // recursive visiting funciton
            fn visit_fn(
                visit_index: usize,
                layer_index: usize,
                mut state: State,
                index_to_inputs: &IndexMap<usize, Vec<usize>>,
                output_indexes: &IndexMap<usize, Vec<usize>>,
            ) -> (usize, State) {
                debug_assert_eq!(state.marks.get(&layer_index), None);

                // if any one of incoming node is not visited, give up and visit later
                for from_layer_index in index_to_inputs[&layer_index].iter().cloned() {
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
                                index_to_inputs,
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
                marks: IndexMap::<usize, Mark>::new(),
                topo_indexes: IndexMap::<usize, usize>::new(),
            };
            let (end_visit_index, new_state) =
                visit_fn(0, 0, state, &index_to_inputs, &output_indexes);
            state = new_state;

            for layer_index in (1..=index_to_config.len()).rev() {
                if let None = state.marks.get(&layer_index) {
                    panic!("the model graph is not connected");
                }
            }

            (0..end_visit_index)
                .map(|topo_index| state.topo_indexes[&topo_index])
                .filter(|layer_index| *layer_index != 0) // exclude input layer
                .collect()
        };

        // compute output channels per layer
        // layer_index -> (in_c?, out_c)
        let index_to_channels: IndexMap<usize, (Option<usize>, usize)> = {
            let init_state: IndexMap<_, _> = {
                let mut state = IndexMap::new();
                state.insert(0, (None, input_channels));
                state
            };

            ordered_layer_indexes
                .iter()
                .cloned()
                .map(|layer_index| (layer_index, &index_to_config[&layer_index]))
                .fold(init_state, |mut channels, (layer_index, layer)| {
                    let from_indexes = &index_to_inputs[&layer_index];

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
                        LayerKind::HeadConv2d { ref anchors, .. } => {
                            debug_assert_eq!(from_indexes.len(), 1);
                            let from_index = from_indexes[0];
                            let in_c = channels[&from_index].1;
                            let out_c = anchors.len() * num_outputs_per_anchor;
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
                })
        };

        // list of exported layer indexes and anchors
        let mut index_to_anchors: IndexMap<usize, Vec<(usize, usize)>> = index_to_config.iter()
            .filter_map(|(layer_index, layer)| {
                match layer.kind {
                    LayerKind::HeadConv2d {ref anchors, ..} => Some((layer_index, anchors.clone())),
                    _ => None
                }
            })
            .map(|(layer_index, anchors)| {
                let out_c = index_to_channels[layer_index].1;
                debug_assert_eq!(out_c, anchors.len() * num_outputs_per_anchor, "the exported layer must have exactly (n_anchors * (n_classes + 5)) output channels");
                (*layer_index, anchors)
            }).collect();

        // build modules for each layer
        // layer_index -> module
        let mut index_to_module: IndexMap<usize, YoloModule> = index_to_config
            .iter()
            .map(|(layer_index, layer_init)| {
                // locals
                let layer_index = *layer_index;
                let LayerInit { kind, .. } = layer_init;

                let from_indexes = &index_to_inputs[&layer_index];
                let (in_c_opt, out_c): (Option<usize>, usize) = index_to_channels[&layer_index];

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
            .collect();

        // construct detection head
        let detection_module = {
            let anchors_list: Vec<Vec<_>> = index_to_anchors
                .iter()
                .map(|(_layer_index, anchors)| {
                    anchors
                        .iter()
                        .cloned()
                        .map(|(height, width)| PixelSize::new(height, width))
                        .collect()
                })
                .collect();
            DetectInit {
                num_classes,
                anchors_list,
            }
            .build(path)
        };

        // construct model
        let layers: Vec<_> = ordered_layer_indexes
            .into_iter()
            .map(|layer_index| {
                let module = index_to_module.remove(&layer_index).unwrap();
                let input_indexes = index_to_inputs.remove(&layer_index).unwrap();
                let anchors_opt = index_to_anchors.remove(&layer_index);

                Layer {
                    layer_index,
                    module,
                    input_indexes,
                    anchors_opt,
                }
            })
            .collect();

        let yolo_model = YoloModel {
            layers,
            detection_module,
        };

        Ok(yolo_model)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerInit {
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
