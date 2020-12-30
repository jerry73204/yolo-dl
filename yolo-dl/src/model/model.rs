use super::{
    config::{LayerInit, LayerKind, YoloInit},
    module::{
        BottleneckCspInit, BottleneckInit, ConvBlockInit, DetectInit, DetectModule, FocusInit,
        SppInit, YoloModule,
    },
};
use crate::common::*;

pub use misc::*;
pub use yolo_model::*;
pub use yolo_output::*;

mod yolo_model {
    use super::*;

    pub use model_config::graph::NodeKey;

    #[derive(Debug)]
    pub struct YoloModel {
        pub(crate) layers: IndexMap<NodeKey, Layer>,
        pub(crate) detection_module: DetectModule,
    }

    impl YoloModel {
        pub fn forward_t(&mut self, xs: &Tensor, train: bool) -> YoloOutput {
            let (_batch_size, _channels, height, width) = xs.size4().unwrap();
            let image_size = PixelSize::new(height, width);
            let mut tmp_tensors: HashMap<NodeKey, Tensor> =
                iter::once((NodeKey(0), xs.shallow_clone())).collect();
            let mut exported_tensors = vec![];

            // run the network
            self.layers.values().for_each(|layer| {
                let Layer {
                    key,
                    ref module,
                    ref input_keys,
                    ref anchors_opt,
                } = *layer;

                let inputs: Vec<_> = input_keys.iter().map(|key| &tmp_tensors[key]).collect();

                let output = module.forward_t(inputs.as_slice(), train);

                if let Some(_anchors) = anchors_opt {
                    exported_tensors.push(output.shallow_clone());
                }
                tmp_tensors.insert(key, output);
            });

            // run detection module
            let exported_tensors: Vec<_> = exported_tensors.iter().collect();
            self.detection_module
                .forward_t(exported_tensors.as_slice(), train, &image_size)
        }

        pub fn from_config<'p, P>(config: &YoloInit, path: P) -> Result<Self>
        where
            P: Borrow<nn::Path<'p>>,
        {
            let path = path.borrow();
            let YoloInit {
                input_channels,
                num_classes,
                depth_multiple,
                width_multiple,
                ref layers,
            } = *config;
            let depth_multiple = depth_multiple.raw();
            let width_multiple = width_multiple.raw();

            ensure!(input_channels > 0, "input_channels must be positive");
            ensure!(num_classes > 0, "num_classes must be positive");
            ensure!(depth_multiple > 0.0, "depth_multiple must be positive");
            ensure!(width_multiple > 0.0, "width_multiple must be positive");
            let num_outputs_per_anchor = num_classes + 5;

            let scale_channel = |channel: usize| -> usize {
                let divisor = 8;
                ((channel as f64 * width_multiple / divisor as f64).ceil() * divisor as f64)
                    as usize
            };

            // annotate each layer with layer index
            // layer_index -> layer_config
            let index_to_config: IndexMap<NodeKey, _> = layers
                .into_iter()
                .enumerate()
                .map(|(index, layer)| {
                    let key = NodeKey(index + 1);
                    (key, layer)
                })
                .collect();

            // compute layer name to index correspondence
            // name -> layer_index
            let name_to_index: IndexMap<&str, NodeKey> = index_to_config
                .iter()
                .filter_map(|(key, layer)| layer.name.as_ref().map(|name| (name.as_str(), *key)))
                .collect();

            // compute input indexes per layer
            // layer_index -> (from_indexes)
            let mut index_to_inputs: IndexMap<NodeKey, Vec<NodeKey>> = index_to_config
                .iter()
                .map(|(key, layer)| {
                    let kind = &layer.kind;

                    let from_indexes = match (kind.from_name(), kind.from_multiple_names()) {
                        (Some(name), None) => {
                            let src_key = *name_to_index
                                .get(name)
                                .expect(&format!(r#"undefined layer name "{}""#, name));
                            vec![src_key]
                        }
                        (None, None) => {
                            let NodeKey(layer_index) = key;
                            let src_key = NodeKey(layer_index - 1);
                            vec![src_key]
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

                    (*key, from_indexes)
                })
                // insert key for input layer
                .chain(iter::once((NodeKey(0), vec![])))
                .collect();

            // topological sort layers
            let sorted_keys: Vec<NodeKey> = {
                let mut graph = DiGraphMap::new();
                index_to_inputs.iter().for_each(|(dst_key, src_keys)| {
                    let dst_key = *dst_key;
                    graph.add_node(dst_key);

                    src_keys.iter().cloned().for_each(|src_key| {
                        graph.add_edge(src_key, dst_key, ());
                    });
                });

                let sorted_keys = petgraph::algo::toposort(&graph, None).map_err(|cycle| {
                    let NodeKey(layer_index) = cycle.node_id();
                    format_err!("cycle detected at layer {}", layer_index)
                })?;

                sorted_keys
            };

            // compute output channels per layer
            // layer_index -> (in_c?, out_c)
            let index_to_channels: IndexMap<NodeKey, (Option<usize>, usize)> = {
                let init_state: IndexMap<_, _> = {
                    let mut state = IndexMap::new();
                    state.insert(NodeKey(0), (None, input_channels));
                    state
                };

                sorted_keys
                    .iter()
                    .cloned()
                    .map(|key| (key, &index_to_config[&key]))
                    .fold(init_state, |mut channels, (key, layer)| {
                        let from_indexes = &index_to_inputs[&key];

                        match layer.kind {
                            LayerKind::Focus { out_c, .. } => {
                                debug_assert_eq!(from_indexes.len(), 1);
                                let from_index = from_indexes[0];
                                let in_c = channels[&from_index].1;
                                let out_c = scale_channel(out_c);
                                channels.insert(key, (Some(in_c), out_c));
                            }
                            LayerKind::ConvBlock { out_c, .. } => {
                                debug_assert_eq!(from_indexes.len(), 1);
                                let from_index = from_indexes[0];
                                let in_c = channels[&from_index].1;
                                let out_c = scale_channel(out_c);
                                channels.insert(key, (Some(in_c), out_c));
                            }
                            LayerKind::Bottleneck { .. } => {
                                debug_assert_eq!(from_indexes.len(), 1);
                                let from_index = from_indexes[0];
                                let in_c = channels[&from_index].1;
                                let out_c = in_c;
                                channels.insert(key, (Some(in_c), out_c));
                            }
                            LayerKind::BottleneckCsp { .. } => {
                                debug_assert_eq!(from_indexes.len(), 1);
                                let from_index = from_indexes[0];
                                let in_c = channels[&from_index].1;
                                let out_c = in_c;
                                channels.insert(key, (Some(in_c), out_c));
                            }
                            LayerKind::Spp { out_c, .. } => {
                                debug_assert_eq!(from_indexes.len(), 1);
                                let from_index = from_indexes[0];
                                let in_c = channels[&from_index].1;
                                let out_c = scale_channel(out_c);
                                channels.insert(key, (Some(in_c), out_c));
                            }
                            LayerKind::HeadConv2d { ref anchors, .. } => {
                                debug_assert_eq!(from_indexes.len(), 1);
                                let from_index = from_indexes[0];
                                let in_c = channels[&from_index].1;
                                let out_c = anchors.len() * num_outputs_per_anchor;
                                channels.insert(key, (Some(in_c), out_c));
                            }
                            LayerKind::Upsample { .. } => {
                                debug_assert_eq!(from_indexes.len(), 1);
                                let from_index = from_indexes[0];
                                let in_c = channels[&from_index].1;
                                let out_c = in_c;
                                channels.insert(key, (Some(in_c), out_c));
                            }
                            LayerKind::Concat { .. } => {
                                let out_c = from_indexes
                                    .iter()
                                    .cloned()
                                    .map(|index| channels[&index].1)
                                    .sum();
                                channels.insert(key, (None, out_c));
                            }
                        }

                        channels
                    })
            };

            // list of exported layer indexes and anchors
            let mut index_to_anchors: IndexMap<NodeKey, Vec<(usize, usize)>> = index_to_config.iter()
                .filter_map(|(key, layer)| {
                    match layer.kind {
                        LayerKind::HeadConv2d {ref anchors, ..} => Some((key, anchors.clone())),
                        _ => None
                    }
                })
                .map(|(key, anchors)| {
                    let out_c = index_to_channels[key].1;
                    debug_assert_eq!(out_c, anchors.len() * num_outputs_per_anchor, "the exported layer must have exactly (n_anchors * (n_classes + 5)) output channels");
                    (*key, anchors)
                }).collect();

            // build modules for each layer
            // key -> module
            let mut index_to_module: IndexMap<NodeKey, YoloModule> = index_to_config
                .iter()
                .map(|(key, layer_init)| {
                    // locals
                    let key = *key;
                    let LayerInit { kind, .. } = layer_init;

                    let src_keys = &index_to_inputs[&key];
                    let (in_c_opt, out_c): (Option<usize>, usize) = index_to_channels[&key];

                    // build layer
                    let module = match *kind {
                        LayerKind::Focus { k, .. } => {
                            debug_assert_eq!(src_keys.len(), 1);
                            let src_key = src_keys[0];
                            let in_c = in_c_opt.unwrap();
                            YoloModule::single(src_key, FocusInit { in_c, out_c, k }.build(path))
                        }
                        LayerKind::ConvBlock { k, s, .. } => {
                            debug_assert_eq!(src_keys.len(), 1);
                            let src_key = src_keys[0];
                            let in_c = in_c_opt.unwrap();

                            YoloModule::single(
                                src_key,
                                ConvBlockInit {
                                    k,
                                    s,
                                    ..ConvBlockInit::new(in_c, out_c)
                                }
                                .build(path),
                            )
                        }
                        LayerKind::Bottleneck { repeat, .. } => {
                            debug_assert_eq!(src_keys.len(), 1);
                            let src_key = src_keys[0];
                            let in_c = in_c_opt.unwrap();
                            let repeat = ((repeat as f64 * depth_multiple).round() as usize).max(1);
                            let bottlenecks = (0..repeat)
                                .into_iter()
                                .map(|_| BottleneckInit::new(in_c, out_c).build(path))
                                .collect::<Vec<_>>();

                            YoloModule::single(src_key, move |xs, train| {
                                bottlenecks
                                    .iter()
                                    .fold(xs.shallow_clone(), |xs, block| block(&xs, train))
                            })
                        }
                        LayerKind::BottleneckCsp {
                            repeat, shortcut, ..
                        } => {
                            debug_assert_eq!(src_keys.len(), 1);
                            let src_key = src_keys[0];
                            let in_c = in_c_opt.unwrap();

                            YoloModule::single(
                                src_key,
                                BottleneckCspInit {
                                    repeat,
                                    shortcut,
                                    ..BottleneckCspInit::new(in_c, out_c)
                                }
                                .build(path),
                            )
                        }
                        LayerKind::Spp { ref ks, .. } => {
                            debug_assert_eq!(src_keys.len(), 1);
                            let src_key = src_keys[0];
                            let in_c = in_c_opt.unwrap();

                            YoloModule::single(
                                src_key,
                                SppInit {
                                    in_c,
                                    out_c,
                                    ks: ks.to_vec(),
                                }
                                .build(path),
                            )
                        }
                        LayerKind::HeadConv2d { k, s, .. } => {
                            debug_assert_eq!(src_keys.len(), 1);
                            let src_key = src_keys[0];
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

                            YoloModule::single(src_key, move |xs, train| xs.apply_t(&conv, train))
                        }
                        LayerKind::Upsample { scale_factor, .. } => {
                            let scale_factor = scale_factor.raw();
                            debug_assert_eq!(src_keys.len(), 1);
                            let src_key = src_keys[0];

                            YoloModule::single(src_key, move |xs, _train| {
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
                            YoloModule::multi(src_keys.to_vec(), move |tensors, _train| {
                                Tensor::cat(tensors, 1)
                            })
                        }
                    };

                    (key, module)
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
            let layers: IndexMap<_, _> = sorted_keys
                .into_iter()
                .map(|key| {
                    let module = index_to_module.remove(&key).unwrap();
                    let input_keys = index_to_inputs.remove(&key).unwrap();
                    let anchors_opt = index_to_anchors.remove(&key);

                    let layer = Layer {
                        key,
                        module,
                        input_keys,
                        anchors_opt,
                    };
                    (key, layer)
                })
                .collect();

            let yolo_model = YoloModel {
                layers,
                detection_module,
            };

            Ok(yolo_model)
        }
    }
}

mod yolo_output {
    use super::*;

    #[derive(Debug, CopyGetters, Getters, TensorLike)]
    pub struct YoloOutput {
        #[getset(get = "pub")]
        pub(crate) image_size: PixelSize<i64>,
        #[getset(get_copy = "pub")]
        pub(crate) batch_size: i64,
        #[getset(get_copy = "pub")]
        pub(crate) num_classes: i64,
        #[tensor_like(copy)]
        #[getset(get_copy = "pub")]
        pub(crate) device: Device,
        #[getset(get = "pub")]
        pub(crate) layer_meta: Vec<LayerMeta>,
        // below tensors have shape [batch, entry, flat] where
        // - flat = \sum_{i is layer_index} (n_anchors_i * feature_height_i * feature_width_i)
        #[getset(get = "pub")]
        pub(crate) cy: Tensor,
        #[getset(get = "pub")]
        pub(crate) cx: Tensor,
        #[getset(get = "pub")]
        pub(crate) height: Tensor,
        #[getset(get = "pub")]
        pub(crate) width: Tensor,
        #[getset(get = "pub")]
        pub(crate) objectness: Tensor,
        #[getset(get = "pub")]
        pub(crate) classification: Tensor,
    }

    impl YoloOutput {
        pub fn cat<T>(outputs: impl IntoIterator<Item = T>, device: Device) -> Result<Self>
        where
            T: Borrow<Self>,
        {
            let (
                image_size_set,
                batch_size_vec,
                num_classes_set,
                layer_meta_set,
                cy_vec,
                cx_vec,
                height_vec,
                width_vec,
                objectness_vec,
                classification_vec,
            ): (
                HashSet<PixelSize<i64>>,
                Vec<i64>,
                HashSet<i64>,
                HashSet<Vec<LayerMeta>>,
                Vec<Tensor>,
                Vec<Tensor>,
                Vec<Tensor>,
                Vec<Tensor>,
                Vec<Tensor>,
                Vec<Tensor>,
            ) = outputs
                .into_iter()
                .map(|output| {
                    let YoloOutput {
                        ref image_size,
                        batch_size,
                        num_classes,
                        ref layer_meta,
                        ref cy,
                        ref cx,
                        ref height,
                        ref width,
                        ref objectness,
                        ref classification,
                        ..
                    } = *output.borrow();

                    (
                        image_size.clone(),
                        batch_size,
                        num_classes,
                        layer_meta.to_owned(),
                        cy.to_device(device),
                        cx.to_device(device),
                        height.to_device(device),
                        width.to_device(device),
                        objectness.to_device(device),
                        classification.to_device(device),
                    )
                })
                .unzip_n();

            let image_size = {
                ensure!(image_size_set.len() == 1, "image_size must be equal");
                image_size_set.into_iter().next().unwrap()
            };
            let num_classes = {
                ensure!(num_classes_set.len() == 1, "num_classes must be equal");
                num_classes_set.into_iter().next().unwrap()
            };
            let layer_meta = {
                ensure!(layer_meta_set.len() == 1, "layer_meta must be equal");
                layer_meta_set.into_iter().next().unwrap()
            };
            let batch_size: i64 = batch_size_vec.into_iter().sum();
            let cy = Tensor::cat(&cy_vec, 0);
            let cx = Tensor::cat(&cx_vec, 0);
            let height = Tensor::cat(&height_vec, 0);
            let width = Tensor::cat(&width_vec, 0);
            let objectness = Tensor::cat(&objectness_vec, 0);
            let classification = Tensor::cat(&classification_vec, 0);

            let flat_index_size: i64 = layer_meta
                .iter()
                .map(|meta| {
                    let LayerMeta {
                        feature_size: GridSize { height, width, .. },
                        ref anchors,
                        ..
                    } = *meta;
                    height * width * anchors.len() as i64
                })
                .sum();

            ensure!(
                cy.size3()? == (batch_size, 1, flat_index_size),
                "invalid cy shape"
            );
            ensure!(
                cx.size3()? == (batch_size, 1, flat_index_size),
                "invalid cx shape"
            );
            ensure!(
                height.size3()? == (batch_size, 1, flat_index_size),
                "invalid height shape"
            );
            ensure!(
                width.size3()? == (batch_size, 1, flat_index_size),
                "invalid width shape"
            );
            ensure!(
                objectness.size3()? == (batch_size, 1, flat_index_size),
                "invalid objectness shape"
            );
            ensure!(
                classification.size3()? == (batch_size, num_classes, flat_index_size),
                "invalid classification shape"
            );

            Ok(Self {
                device,
                image_size,
                num_classes,
                layer_meta,
                batch_size,
                cy,
                cx,
                height,
                width,
                objectness,
                classification,
            })
        }

        pub fn flat_to_instance_index(
            &self,
            batch_index: usize,
            flat_index: i64,
        ) -> Option<InstanceIndex> {
            let Self { batch_size, .. } = *self;

            if batch_index as i64 >= batch_size || flat_index < 0 {
                return None;
            }

            let (
                layer_index,
                LayerMeta {
                    feature_size:
                        GridSize {
                            height: feature_h,
                            width: feature_w,
                            ..
                        },
                    anchors,
                    flat_index_range,
                    ..
                },
            ) = self
                .layer_meta
                .iter()
                .enumerate()
                .find(|(_layer_index, meta)| flat_index < meta.flat_index_range.end)?;

            // flat_index = begin_flat_index + col + row * (width + anchor_index * height)
            let remainder = flat_index - flat_index_range.start;
            let grid_col = remainder % feature_w;
            let grid_row = remainder / feature_w % feature_h;
            let anchor_index = remainder / feature_w / feature_h;

            if anchor_index >= anchors.len() as i64 {
                return None;
            }

            Some(InstanceIndex {
                batch_index,
                layer_index,
                anchor_index,
                grid_row,
                grid_col,
            })
        }

        pub fn instance_to_flat_index(&self, instance_index: &InstanceIndex) -> i64 {
            let InstanceIndex {
                layer_index,
                anchor_index,
                grid_row,
                grid_col,
                ..
            } = *instance_index;

            let LayerMeta {
                ref flat_index_range,
                feature_size: GridSize { height, width, .. },
                ..
            } = self.layer_meta[layer_index];

            let flat_index =
                flat_index_range.start + grid_col + width * (grid_row + height * anchor_index);

            flat_index
        }

        pub fn feature_maps(&self) -> Vec<FeatureMap> {
            let Self {
                batch_size,
                num_classes,
                ref layer_meta,
                ..
            } = *self;

            let feature_maps = layer_meta
                .iter()
                .enumerate()
                .map(|(_layer_index, meta)| {
                    let LayerMeta {
                        feature_size:
                            GridSize {
                                height: feature_h,
                                width: feature_w,
                                ..
                            },
                        ref anchors,
                        ref flat_index_range,
                        ..
                    } = *meta;
                    let num_anchors = anchors.len() as i64;

                    let cy_map = self.cy.i((.., .., flat_index_range.clone())).view([
                        batch_size,
                        1,
                        num_anchors,
                        feature_h,
                        feature_w,
                    ]);
                    let cx_map = self.cx.i((.., .., flat_index_range.clone())).view([
                        batch_size,
                        1,
                        num_anchors,
                        feature_h,
                        feature_w,
                    ]);
                    let h_map = self.height.i((.., .., flat_index_range.clone())).view([
                        batch_size,
                        1,
                        num_anchors,
                        feature_h,
                        feature_w,
                    ]);
                    let w_map = self.width.i((.., .., flat_index_range.clone())).view([
                        batch_size,
                        1,
                        num_anchors,
                        feature_h,
                        feature_w,
                    ]);
                    let objectness_map = self
                        .objectness
                        .i((.., .., flat_index_range.clone()))
                        .view([batch_size, 1, num_anchors, feature_h, feature_w]);
                    let classification_map = self
                        .classification
                        .i((.., .., flat_index_range.clone()))
                        .view([batch_size, num_classes, num_anchors, feature_h, feature_w]);

                    FeatureMap {
                        cy: cy_map,
                        cx: cx_map,
                        h: h_map,
                        w: w_map,
                        objectness: objectness_map,
                        classification: classification_map,
                    }
                })
                .collect_vec();

            feature_maps
        }
    }
}

mod misc {
    use super::*;

    #[derive(Debug)]
    pub struct Layer {
        pub(crate) key: NodeKey,
        pub(crate) module: YoloModule,
        pub(crate) input_keys: Vec<NodeKey>,
        pub(crate) anchors_opt: Option<Vec<(usize, usize)>>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, TensorLike)]
    pub struct LayerMeta {
        /// feature map size in grid units
        #[tensor_like(clone)]
        pub feature_size: GridSize<i64>,
        /// per grid size in pixel units
        #[tensor_like(clone)]
        pub grid_size: PixelSize<R64>,
        /// Anchros (height, width) in grid units
        #[tensor_like(clone)]
        pub anchors: Vec<GridSize<R64>>,
        #[tensor_like(clone)]
        pub flat_index_range: Range<i64>,
    }

    #[derive(Debug, TensorLike)]
    pub struct FeatureMap {
        // tensors have shape [batch, entry, anchor, height, width]
        pub cy: Tensor,
        pub cx: Tensor,
        pub h: Tensor,
        pub w: Tensor,
        pub objectness: Tensor,
        pub classification: Tensor,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, TensorLike)]
    pub struct InstanceIndex {
        pub batch_index: usize,
        pub layer_index: usize,
        pub anchor_index: i64,
        pub grid_row: i64,
        pub grid_col: i64,
    }
}
