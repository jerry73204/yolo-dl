use super::*;
use crate::common::*;

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
        let index_to_config: HashMap<usize, _> = layers
            .into_iter()
            .enumerate()
            .map(|(index, layer)| {
                let layer_index = index + 1;
                (layer_index, layer)
            })
            .collect();

        // compute layer name to index correspondence
        // name -> layer_index
        let name_to_index: HashMap<&str, usize> = index_to_config
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
        let mut index_to_inputs: HashMap<usize, Vec<usize>> = index_to_config
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
                        .collect::<Vec<_>>(),
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
                marks: HashMap<usize, Mark>,
                topo_indexes: HashMap<usize, usize>,
            }

            // list directed edges among nodes
            // layer_index -> [output_indexes]
            let output_indexes = index_to_inputs
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
                index_to_inputs: &HashMap<usize, Vec<usize>>,
                output_indexes: &HashMap<usize, Vec<usize>>,
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
                marks: HashMap::<usize, Mark>::new(),
                topo_indexes: HashMap::<usize, usize>::new(),
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
        let index_to_channels: HashMap<usize, (Option<usize>, usize)> = {
            let init_state: HashMap<_, _> = {
                let mut state = HashMap::new();
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
        let mut index_to_anchors: HashMap<usize, Vec<(usize, usize)>> = index_to_config.iter()
            .filter_map(|(layer_index, layer)| {
                match layer.kind {
                    LayerKind::HeadConv2d {ref anchors, ..} => Some((layer_index, anchors.clone())),
                    _ => None
                }
            })
            .map(|(layer_index, anchors)| {
                let out_c = index_to_channels[&layer_index].1;
                debug_assert_eq!(out_c, anchors.len() * num_outputs_per_anchor, "the exported layer must have exactly (n_anchros * (n_classes + 5)) output channels");
                (*layer_index, anchors)
            }).collect();

        // build modules for each layer
        // layer_index -> module
        let mut index_to_module: HashMap<usize, YoloModule> = index_to_config
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
            let anchors_list: Vec<_> = index_to_anchors
                .iter()
                .map(|(_layer_index, anchros)| anchros.to_vec())
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

#[derive(Debug)]
pub struct YoloModel {
    layers: Vec<Layer>,
    detection_module: DetectModule,
}

impl YoloModel {
    pub fn forward_t(&self, xs: &Tensor, train: bool) -> YoloOutput {
        let (_batch_size, _channels, height, width) = xs.size4().unwrap();
        let mut tmp_tensors: HashMap<usize, Tensor> = iter::once((0, xs.shallow_clone())).collect();
        let mut exported_tensors = vec![];

        // run the network
        self.layers.iter().for_each(|layer| {
            let Layer {
                layer_index,
                ref module,
                ref input_indexes,
                ref anchors_opt,
            } = *layer;

            let inputs: Vec<_> = input_indexes
                .iter()
                .map(|from_index| &tmp_tensors[from_index])
                .collect();

            let output = module.forward_t(inputs.as_slice(), train);

            if let Some(_anchors) = anchors_opt {
                exported_tensors.push(output.shallow_clone());
            }
            tmp_tensors.insert(layer_index, output);
        });

        // run detection module
        let exported_tensors: Vec<_> = exported_tensors.iter().collect();
        self.detection_module
            .forward_t(exported_tensors.as_slice(), train, height, width)
    }
}

#[derive(Debug)]
struct Layer {
    layer_index: usize,
    module: YoloModule,
    input_indexes: Vec<usize>,
    anchors_opt: Option<Vec<(usize, usize)>>,
}

#[derive(Debug, TensorLike)]
pub struct YoloOutput {
    pub(crate) image_height: i64,
    pub(crate) image_width: i64,
    pub(crate) batch_size: i64,
    pub(crate) num_classes: i64,
    #[tensor_like(copy)]
    pub(crate) device: Device,
    /// Grid indexes tensor of shape \[bsize, num_grids, 3\]
    ///
    /// The last dimension represents (anchor_array_index, grid_row, grid_col)
    /// The anchor_array_index is the index of anchor_sizes field.
    pub(crate) grid_indexes: Tensor,
    /// Compount output tensor of shape \[bsize, num_grids, 5 + num_classes\]
    ///
    /// The last dimension represents (cy, cx, h, w, objectness, classifications..)
    pub(crate) outputs: Tensor,
    /// Tensor of anchor box sizes of shape [n_anchors, 2], where n_anchors is indexed by anchor_array_index.
    ///
    /// The last dimension represents (height, width) in grid units
    pub(crate) anchor_sizes: Tensor,
    pub(crate) feature_info: Vec<FeatureInfo>,
}

impl YoloOutput {
    pub fn image_height(&self) -> i64 {
        self.image_height
    }

    pub fn image_width(&self) -> i64 {
        self.image_width
    }

    pub fn anchor_indexes(&self) -> Tensor {
        self.grid_indexes.i((.., .., 0..1))
    }

    pub fn grid_rows(&self) -> Tensor {
        self.grid_indexes.i((.., .., 1..2))
    }

    pub fn grid_cols(&self) -> Tensor {
        self.grid_indexes.i((.., .., 2..3))
    }

    pub fn bbox_cy(&self) -> Tensor {
        self.outputs.i((.., .., 0..1))
    }

    pub fn bbox_cx(&self) -> Tensor {
        self.outputs.i((.., .., 1..2))
    }

    pub fn bbox_h(&self) -> Tensor {
        self.outputs.i((.., .., 2..3))
    }

    pub fn bbox_w(&self) -> Tensor {
        self.outputs.i((.., .., 3..4))
    }

    pub fn objectnesses(&self) -> Tensor {
        self.outputs.i((.., .., 4..5))
    }

    pub fn classifications(&self) -> Tensor {
        self.outputs.i((.., .., 5..))
    }

    pub fn batch_size(&self) -> i64 {
        self.batch_size
    }
}

#[derive(Debug, TensorLike)]
pub struct FeatureInfo {
    /// Feature map height in grid units
    pub feature_height: i64,
    /// Feature map width in grid units
    pub feature_width: i64,
    /// Per-grid height in pixels
    pub per_grid_height: f64,
    /// Per-grid width in pixels
    pub per_grid_width: f64,
    /// Anchros (height, width) in grid units
    pub anchors: Vec<(f64, f64)>,
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
    pub cycxhw: Tensor,
    pub objectness: Tensor,
    pub classification: Tensor,
}
