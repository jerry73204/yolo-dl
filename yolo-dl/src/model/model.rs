use super::{
    config::{LayerInit, LayerKind, YoloInit},
    module::{
        BottleneckCspInit, BottleneckInit, Concat2D, ConvBlockInit, ConvBn2DInit, DarkCsp2DInit,
        Detect2DInit, DetectInit, FocusInit, Input, MergeDetect2D, Module, ModuleInput,
        ModuleOutput, SppCsp2DInit, SppInit, Sum2D, UpSample2D,
    },
};
use crate::common::*;

pub use misc::*;
pub use yolo_model::*;
pub use yolo_output::*;

pub use model_config::graph::{InputKeys, NodeKey};

mod model {
    use super::*;

    pub struct Model {
        nodes: IndexMap<NodeKey, Layer>,
    }

    impl Model {
        pub fn from_graph<'p>(
            orig_graph: &'_ model_config::graph::Graph,
            path: impl Borrow<nn::Path<'p>>,
        ) -> Result<Self> {
            use model_config::{config, graph};
            let path = path.borrow();
            let orig_nodes = orig_graph.nodes();

            let nodes: IndexMap<_, _> = orig_nodes
                .iter()
                .map(|(&key, node)| -> Result<_> {
                    let graph::Node {
                        input_keys, config, ..
                    } = node;

                    let module = match *config {
                        config::Module::Input(_) => Module::Input(Input::new()),
                        config::Module::ConvBn2D(config::ConvBn2D {
                            c,
                            k,
                            s,
                            p,
                            d,
                            g,
                            act,
                            bn,
                            ..
                        }) => {
                            let src_key = input_keys.single().unwrap();
                            let [_b, in_c, _h, _w] = orig_nodes[&src_key]
                                .output_shape
                                .as_tensor()
                                .unwrap()
                                .size4()
                                .unwrap();
                            let in_c = in_c.size().unwrap();

                            Module::ConvBn2D(
                                ConvBn2DInit {
                                    in_c,
                                    out_c: c,
                                    k,
                                    s,
                                    p,
                                    d,
                                    g,
                                    activation: act,
                                    batch_norm: bn,
                                }
                                .build(path),
                            )
                        }
                        config::Module::UpSample2D(config::UpSample2D { scale, .. }) => {
                            Module::UpSample2D(UpSample2D::new(scale.raw())?)
                        }
                        config::Module::DarkCsp2D(config::DarkCsp2D {
                            c,
                            repeat,
                            shortcut,
                            c_mul,
                            ..
                        }) => {
                            let src_key = input_keys.single().unwrap();
                            let [_b, in_c, _h, _w] = orig_nodes[&src_key]
                                .output_shape
                                .as_tensor()
                                .unwrap()
                                .size4()
                                .unwrap();
                            let in_c = in_c.size().unwrap();

                            Module::DarkCsp2D(
                                DarkCsp2DInit {
                                    in_c,
                                    out_c: c,
                                    repeat,
                                    shortcut,
                                    c_mul,
                                }
                                .build(path),
                            )
                        }
                        config::Module::SppCsp2D(config::SppCsp2D {
                            c, ref k, c_mul, ..
                        }) => {
                            let src_key = input_keys.single().unwrap();
                            let [_b, in_c, _h, _w] = orig_nodes[&src_key]
                                .output_shape
                                .as_tensor()
                                .unwrap()
                                .size4()
                                .unwrap();
                            let in_c = in_c.size().unwrap();

                            Module::SppCsp2D(
                                SppCsp2DInit {
                                    in_c,
                                    out_c: c,
                                    k: k.to_owned(),
                                    c_mul,
                                }
                                .build(path),
                            )
                        }
                        config::Module::Sum2D(_) => Module::Sum2D(Sum2D),
                        config::Module::Concat2D(_) => Module::Concat2D(Concat2D),
                        config::Module::Detect2D(config::Detect2D {
                            classes,
                            ref anchors,
                            ..
                        }) => {
                            let anchors: Vec<_> = anchors
                                .iter()
                                .map(|size| -> Result<_> {
                                    let config::Size { h, w } = *size;
                                    let size = RatioSize::new(h.try_into()?, w.try_into()?);
                                    Ok(size)
                                })
                                .try_collect()?;

                            Module::Detect2D(
                                Detect2DInit {
                                    num_classes: classes,
                                    anchors,
                                }
                                .build(path),
                            )
                        }
                        config::Module::GroupRef(_) => unreachable!(),
                        config::Module::MergeDetect2D(_) => {
                            Module::MergeDetect2D(MergeDetect2D::new())
                        }
                    };

                    let layer = Layer {
                        key,
                        input_keys: input_keys.to_owned(),
                        module,
                    };

                    Ok((key, layer))
                })
                .try_collect()?;

            Ok(Self { nodes })
        }
    }
}

mod yolo_model {
    use super::*;

    #[derive(Debug)]
    pub struct YoloModel {
        pub(crate) layers: IndexMap<NodeKey, Layer>,
        pub(crate) output_key: NodeKey,
    }

    impl YoloModel {
        pub fn forward_t(&mut self, input: &Tensor, train: bool) -> Result<ModuleOutput> {
            let Self {
                ref mut layers,
                output_key,
            } = *self;
            let (_batch_size, _channels, height, width) = input.size4().unwrap();
            let image_size = PixelSize::new(height, width);
            let mut module_outputs: HashMap<NodeKey, ModuleOutput> = HashMap::new();
            let mut input = Some(input); // it makes sure the input is consumed at most once

            // run the network
            layers.values_mut().try_for_each(|layer| -> Result<_> {
                let Layer {
                    key,
                    ref mut module,
                    ref input_keys,
                } = *layer;

                let module_input: ModuleInput = match input_keys {
                    InputKeys::None => ModuleInput::None,
                    InputKeys::PlaceHolder => input.take().unwrap().into(),
                    InputKeys::Single(src_key) => (&module_outputs[src_key]).try_into()?,
                    InputKeys::Indexed(src_keys) => {
                        let inputs: Vec<_> = src_keys
                            .iter()
                            .map(|src_key| &module_outputs[src_key])
                            .collect();
                        inputs.as_slice().try_into()?
                    }
                };
                let module_output = module.forward_t(module_input, train, &image_size)?;
                module_outputs.insert(key, module_output);
                Ok(())
            })?;

            debug_assert!(input.is_none());

            // extract output
            let output = module_outputs.remove(&output_key).unwrap();

            Ok(output)
        }

        pub fn from_config<'p, P>(config: &YoloInit, path: P) -> Result<Self>
        where
            P: Borrow<nn::Path<'p>>,
        {
            enum ConfigKind<'a> {
                Input,
                Layer(&'a LayerInit),
                Detection,
            }

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

            // index the layers
            let (index_to_config, index_to_prev_key) = {
                let mut index_to_config = IndexMap::new();
                let mut index_to_prev_key = IndexMap::new();
                let mut key_enumerator = iter::repeat(())
                    .enumerate()
                    .map(|(index, ())| NodeKey(index));

                let input_key = key_enumerator.next().unwrap();
                index_to_config.insert(input_key, ConfigKind::Input);

                index_to_config.insert(key_enumerator.next().unwrap(), ConfigKind::Detection);

                layers
                    .into_iter()
                    .map(|layer| {
                        let key = key_enumerator.next().unwrap();
                        let config = ConfigKind::Layer(layer);
                        (key, config)
                    })
                    .scan(input_key, |prev_key, (key, config)| {
                        let infer_prev_key = *prev_key;
                        *prev_key = key;
                        Some((key, infer_prev_key, config))
                    })
                    .for_each(|(key, infer_prev_key, config)| {
                        index_to_config.insert(key, config);
                        index_to_prev_key.insert(key, infer_prev_key);
                    });

                (index_to_config, index_to_prev_key)
            };

            // compute layer name to index correspondence
            // name -> layer_index
            let name_to_index: IndexMap<&str, NodeKey> = index_to_config
                .iter()
                .filter_map(|(key, config)| match config {
                    ConfigKind::Layer(layer) => {
                        layer.name.as_ref().map(|name| (name.as_str(), *key))
                    }
                    _ => None,
                })
                .collect();

            // compute input indexes per layer
            // layer_index -> (from_indexes)
            let mut index_to_inputs: IndexMap<NodeKey, InputKeys> = index_to_config
                .iter()
                .map(|(&key, config)| {
                    let src_keys = match config {
                        ConfigKind::Input => InputKeys::PlaceHolder,
                        ConfigKind::Detection => {
                            let src_keys: Vec<_> = index_to_config
                                .iter()
                                .filter_map(|(&key, config)| match config {
                                    ConfigKind::Layer(LayerInit {
                                        kind: LayerKind::HeadConv2d { .. },
                                        ..
                                    }) => Some(key),
                                    _ => None,
                                })
                                .collect();

                            InputKeys::Indexed(src_keys)
                        }
                        ConfigKind::Layer(layer) => {
                            let kind = &layer.kind;
                            let src_keys = match (kind.from_name(), kind.from_multiple_names()) {
                                (Some(name), None) => {
                                    let src_key = *name_to_index
                                        .get(name)
                                        .expect(&format!(r#"undefined layer name "{}""#, name));
                                    InputKeys::Single(src_key)
                                }
                                (None, None) => {
                                    // inferred as previous layer
                                    let src_key = index_to_prev_key[&key];
                                    InputKeys::Single(src_key)
                                }
                                (None, Some(names)) => {
                                    let src_keys: Vec<_> = names
                                        .iter()
                                        .map(|name| {
                                            *name_to_index.get(name.as_str()).expect(&format!(
                                                r#"undefined layer name "{}""#,
                                                name
                                            ))
                                        })
                                        .collect();
                                    InputKeys::Indexed(src_keys)
                                }
                                _ => unreachable!("please report bug"),
                            };

                            src_keys
                        }
                    };

                    (key, src_keys)
                })
                .collect();

            // topological sort layers
            let sorted_keys: Vec<NodeKey> = {
                let mut graph = DiGraphMap::new();
                index_to_inputs.iter().for_each(|(dst_key, src_keys)| {
                    let dst_key = *dst_key;
                    graph.add_node(dst_key);

                    src_keys.iter().for_each(|src_key| {
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
            let index_to_out_channels: IndexMap<NodeKey, usize> =
                sorted_keys
                    .iter()
                    .cloned()
                    .fold(IndexMap::new(), |mut channels_map, key| {
                        let from_indexes = &index_to_inputs[&key];
                        let config = &index_to_config[&key];

                        let out_c = match config {
                            ConfigKind::Layer(layer) => match layer.kind {
                                LayerKind::Focus { out_c, .. } => {
                                    let out_c = scale_channel(out_c);
                                    Some(out_c)
                                }
                                LayerKind::ConvBlock { out_c, .. } => {
                                    let out_c = scale_channel(out_c);
                                    Some(out_c)
                                }
                                LayerKind::Bottleneck { .. } => {
                                    let from_index = from_indexes.single().unwrap();
                                    let in_c = channels_map[&from_index];
                                    let out_c = in_c;
                                    Some(out_c)
                                }
                                LayerKind::BottleneckCsp { .. } => {
                                    let from_index = from_indexes.single().unwrap();
                                    let in_c = channels_map[&from_index];
                                    let out_c = in_c;
                                    Some(out_c)
                                }
                                LayerKind::Spp { out_c, .. } => {
                                    let out_c = scale_channel(out_c);
                                    Some(out_c)
                                }
                                LayerKind::HeadConv2d { ref anchors, .. } => {
                                    let out_c = anchors.len() * num_outputs_per_anchor;
                                    Some(out_c)
                                }
                                LayerKind::Upsample { .. } => {
                                    let from_index = from_indexes.single().unwrap();
                                    let in_c = channels_map[&from_index];
                                    let out_c = in_c;
                                    Some(out_c)
                                }
                                LayerKind::Concat { .. } => {
                                    let out_c = from_indexes
                                        .indexed()
                                        .unwrap()
                                        .iter()
                                        .cloned()
                                        .map(|index| channels_map[&index])
                                        .sum();
                                    Some(out_c)
                                }
                            },
                            ConfigKind::Input => Some(input_channels),
                            ConfigKind::Detection => {
                                from_indexes.indexed().unwrap().iter().for_each(|src_key| {
                                    let anchors = match index_to_config[src_key] {
                                        ConfigKind::Layer(LayerInit {
                                            kind: LayerKind::HeadConv2d { anchors, .. },
                                            ..
                                        }) => anchors,
                                        _ => unreachable!(),
                                    };
                                    let out_c = channels_map[src_key];
                                    debug_assert_eq!(out_c, anchors.len() * num_outputs_per_anchor);
                                });

                                // detect module must be final layer, no output channel provided here
                                None
                            }
                        };

                        if let Some(out_c) = out_c {
                            channels_map.insert(key, out_c);
                        }
                        channels_map
                    });

            // build modules for each layer
            // key -> module
            let mut index_to_module: IndexMap<NodeKey, Module> = sorted_keys
                .iter()
                .cloned()
                .map(|key| {
                    let module = {
                        let config = &index_to_config[&key];

                        match config {
                            ConfigKind::Layer(layer) => {
                                let LayerInit { kind, .. } = layer;
                                let src_keys = &index_to_inputs[&key];
                                let out_c = index_to_out_channels[&key];

                                // build layer
                                let module = match *kind {
                                    LayerKind::Focus { k, .. } => {
                                        let src_key = src_keys.single().unwrap();
                                        let in_c = index_to_out_channels[&src_key];
                                        Module::FnSingle(FocusInit { in_c, out_c, k }.build(path))
                                    }
                                    LayerKind::ConvBlock { k, s, .. } => {
                                        let src_key = src_keys.single().unwrap();
                                        let in_c = index_to_out_channels[&src_key];

                                        Module::FnSingle(
                                            ConvBlockInit {
                                                k,
                                                s,
                                                ..ConvBlockInit::new(in_c, out_c)
                                            }
                                            .build(path),
                                        )
                                    }
                                    LayerKind::Bottleneck { repeat, .. } => {
                                        let src_key = src_keys.single().unwrap();
                                        let in_c = index_to_out_channels[&src_key];
                                        let repeat = ((repeat as f64 * depth_multiple).round()
                                            as usize)
                                            .max(1);
                                        let bottlenecks = (0..repeat)
                                            .into_iter()
                                            .map(|_| BottleneckInit::new(in_c, out_c).build(path))
                                            .collect::<Vec<_>>();

                                        Module::FnSingle(Box::new(move |xs: &Tensor, train| {
                                            bottlenecks
                                                .iter()
                                                .fold(xs.shallow_clone(), |xs, block| {
                                                    block(&xs, train)
                                                })
                                        }))
                                    }
                                    LayerKind::BottleneckCsp {
                                        repeat, shortcut, ..
                                    } => {
                                        let src_key = src_keys.single().unwrap();
                                        let in_c = index_to_out_channels[&src_key];

                                        Module::FnSingle(
                                            BottleneckCspInit {
                                                repeat,
                                                shortcut,
                                                ..BottleneckCspInit::new(in_c, out_c)
                                            }
                                            .build(path),
                                        )
                                    }
                                    LayerKind::Spp { ref ks, .. } => {
                                        let src_key = src_keys.single().unwrap();
                                        let in_c = index_to_out_channels[&src_key];

                                        Module::FnSingle(
                                            SppInit {
                                                in_c,
                                                out_c,
                                                ks: ks.to_vec(),
                                            }
                                            .build(path),
                                        )
                                    }
                                    LayerKind::HeadConv2d { k, s, .. } => {
                                        let src_key = src_keys.single().unwrap();
                                        let in_c = index_to_out_channels[&src_key];
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

                                        Module::FnSingle(Box::new(move |xs, train| {
                                            xs.apply_t(&conv, train)
                                        }))
                                    }
                                    LayerKind::Upsample { scale_factor, .. } => {
                                        let scale_factor = scale_factor.raw();

                                        Module::FnSingle(Box::new(move |xs, _train| {
                                            let (height, width) = match xs.size().as_slice() {
                                                &[_bsize, _channels, height, width] => {
                                                    (height, width)
                                                }
                                                _ => unreachable!(),
                                            };

                                            let new_height = (height as f64 * scale_factor) as i64;
                                            let new_width = (width as f64 * scale_factor) as i64;

                                            xs.upsample_nearest2d(
                                                &[new_height, new_width],
                                                Some(scale_factor),
                                                Some(scale_factor),
                                            )
                                        }))
                                    }
                                    LayerKind::Concat { .. } => {
                                        Module::FnIndexed(Box::new(move |tensors, _train| {
                                            Tensor::cat(tensors, 1)
                                        }))
                                    }
                                };

                                module
                            }
                            ConfigKind::Input => {
                                debug_assert!(key == NodeKey(0));
                                Module::Input(Input::new())
                            }
                            ConfigKind::Detection => {
                                let src_keys = &index_to_inputs[&key];
                                let anchors_list: Vec<Vec<_>> = src_keys
                                    .indexed()
                                    .unwrap()
                                    .iter()
                                    .map(|src_key| {
                                        let anchors = match index_to_config[src_key] {
                                            ConfigKind::Layer(LayerInit {
                                                kind: LayerKind::HeadConv2d { anchors, .. },
                                                ..
                                            }) => anchors,
                                            _ => unreachable!(),
                                        };

                                        let anchors: Vec<_> = anchors
                                            .iter()
                                            .cloned()
                                            .map(|(height, width)| PixelSize::new(height, width))
                                            .collect();

                                        anchors
                                    })
                                    .collect();

                                let module = DetectInit {
                                    num_classes,
                                    anchors_list,
                                }
                                .build(path);

                                Module::Detect(module)
                            }
                        }
                    };
                    (key, module)
                })
                .collect();

            // construct model
            let layers: IndexMap<_, _> = sorted_keys
                .into_iter()
                .map(|key| {
                    let module = index_to_module.remove(&key).unwrap();
                    let input_keys = index_to_inputs.remove(&key).unwrap();

                    let layer = Layer {
                        key,
                        module,
                        input_keys,
                    };
                    (key, layer)
                })
                .collect();

            let output_key = layers
                .iter()
                .find_map(|(&key, layer)| match layer.module {
                    Module::Detect(_) => Some(key),
                    _ => None,
                })
                .ok_or_else(|| format_err!("TODO"))?;

            let yolo_model = YoloModel { output_key, layers };

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
        pub(crate) module: Module,
        pub(crate) input_keys: InputKeys,
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
