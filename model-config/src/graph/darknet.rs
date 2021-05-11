use super::graph::*;
use crate::{
    common::*,
    config::{self, Module, ShapeOutput},
};
use darknet_config::config as dark_cfg;

use dark_input_keys::*;
use dark_node_key::*;

mod dark_node_key {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
    pub enum DarkNodeKey {
        Input,
        Index(usize),
    }

    impl DarkNodeKey {
        pub fn index(&self) -> Option<usize> {
            match *self {
                Self::Index(index) => Some(index),
                Self::Input => None,
            }
        }
    }

    impl Display for DarkNodeKey {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
            match *self {
                DarkNodeKey::Input => formatter.write_str("[input]"),
                DarkNodeKey::Index(index) => formatter.write_str(&index.to_string()),
            }
        }
    }

    impl From<DarkNodeKey> for NodeKey {
        fn from(from: DarkNodeKey) -> Self {
            match from {
                DarkNodeKey::Input => Self(0),
                DarkNodeKey::Index(index) => Self(index + 1),
            }
        }
    }

    impl From<NodeKey> for DarkNodeKey {
        fn from(from: NodeKey) -> Self {
            match from.0 {
                0 => Self::Input,
                key => Self::Index(key - 1),
            }
        }
    }
}

mod dark_input_keys {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub enum DarkInputKeys {
        None,
        PlaceHolder,
        Single(DarkNodeKey),
        Indexed(Vec<DarkNodeKey>),
    }

    impl From<DarkInputKeys> for InputKeys {
        fn from(from: DarkInputKeys) -> Self {
            match from {
                DarkInputKeys::None => Self::None,
                DarkInputKeys::PlaceHolder => Self::PlaceHolder,
                DarkInputKeys::Single(key) => Self::Single(key.into()),
                DarkInputKeys::Indexed(keys) => {
                    Self::Indexed(keys.into_iter().map(Into::into).collect())
                }
            }
        }
    }

    impl DarkInputKeys {
        pub fn iter(&self) -> impl Iterator<Item = DarkNodeKey> {
            let iter: Box<dyn Iterator<Item = DarkNodeKey>> = match *self {
                Self::None => Box::new(iter::empty()),
                Self::PlaceHolder => Box::new(iter::empty()),
                Self::Single(key) => Box::new(iter::once(key)),
                Self::Indexed(ref keys) => Box::new(keys.clone().into_iter()),
            };
            iter
        }

        pub fn single(&self) -> Option<DarkNodeKey> {
            match *self {
                Self::Single(key) => Some(key),
                _ => None,
            }
        }

        pub fn indexed(&self) -> Option<&[DarkNodeKey]> {
            match self {
                Self::Indexed(keys) => Some(keys.as_slice()),
                _ => None,
            }
        }
    }
}

impl Graph {
    pub fn from_darknet(config: &dark_cfg::Darknet) -> Result<Self> {
        // load config file
        let dark_cfg::Darknet {
            net:
                dark_cfg::Net {
                    input_size: model_input_shape,
                    ..
                },
            ref layers,
        } = *config;

        // compute from indexes per layer
        let input_keys_map: IndexMap<DarkNodeKey, _> =
            iter::once(Ok((DarkNodeKey::Input, DarkInputKeys::None)))
                .chain(
                    layers
                        .iter()
                        .enumerate()
                        .map(|(layer_index, layer_config)| -> Result<_> {
                            let from_indexes = match layer_config {
                                dark_cfg::Layer::Convolutional(_)
                                | dark_cfg::Layer::Connected(_)
                                | dark_cfg::Layer::BatchNorm(_)
                                | dark_cfg::Layer::MaxPool(_)
                                | dark_cfg::Layer::UpSample(_)
                                | dark_cfg::Layer::Dropout(_)
                                | dark_cfg::Layer::Softmax(_)
                                | dark_cfg::Layer::GaussianYolo(_)
                                | dark_cfg::Layer::Yolo(_) => {
                                    if layer_index == 0 {
                                        DarkInputKeys::Single(DarkNodeKey::Input)
                                    } else {
                                        DarkInputKeys::Single(DarkNodeKey::Index(layer_index - 1))
                                    }
                                }
                                dark_cfg::Layer::Shortcut(conf) => {
                                    let from = &conf.from;
                                    let first_index = if layer_index == 0 {
                                        DarkNodeKey::Input
                                    } else {
                                        DarkNodeKey::Index(layer_index - 1)
                                    };

                                    let from_indexes: Vec<_> = iter::once(Ok(first_index))
                                        .chain(from.iter().map(|index| -> Result<_> {
                                            let index = index.to_absolute(layer_index).ok_or_else(
                                                || format_err!("invalid layer index {}", index),
                                            )?;
                                            Ok(DarkNodeKey::Index(index))
                                        }))
                                        .try_collect()?;

                                    ensure!(
                                        from_indexes.len() == from.len() + 1,
                                        "from must not contain the index to previous layer"
                                    );

                                    DarkInputKeys::Indexed(from_indexes)
                                }
                                dark_cfg::Layer::Route(conf) => {
                                    let from_indexes: Vec<_> = conf
                                        .layers
                                        .iter()
                                        .map(|&index| -> Result<_> {
                                            let index = index.to_absolute(layer_index).ok_or_else(
                                                || format_err!("invalid layer index {}", index),
                                            )?;
                                            Ok(DarkNodeKey::Index(index))
                                        })
                                        .try_collect()?;
                                    DarkInputKeys::Indexed(from_indexes)
                                }
                                _ => unimplemented!(),
                            };

                            let key = DarkNodeKey::Index(layer_index);
                            Ok((key, from_indexes))
                        }),
                )
                .try_collect()?;

        // topological sort
        let sorted_node_keys: Vec<DarkNodeKey> = {
            let graph = {
                let mut graph = DiGraphMap::<DarkNodeKey, ()>::new();
                input_keys_map.iter().for_each(|(&key, from_indexes)| {
                    graph.add_node(key);
                    from_indexes.iter().for_each(|from_key| {
                        graph.add_edge(from_key, key, ());
                    });
                });
                graph
            };

            let sorted_node_keys = petgraph::algo::toposort(&graph, None)
                .map_err(|cycle| format_err!("cycle detected at layer {:?}", cycle.node_id()))?;

            sorted_node_keys
        };

        let layer_configs: IndexMap<DarkNodeKey, _> = sorted_node_keys
            .iter()
            .filter_map(|&key| Some((key, &layers[key.index()?])))
            .collect();

        // compute shapes
        let output_shape: HashMap<DarkNodeKey, ShapeOutput> = sorted_node_keys.iter().try_fold(
            HashMap::new(),
            |mut collected: HashMap<DarkNodeKey, ShapeOutput>, &key| -> Result<_> {
                (|| -> Result<_> {
                    let output_shape: ShapeOutput = match key {
                        DarkNodeKey::Input => match model_input_shape {
                            dark_cfg::Shape::Dim3([h, w, c]) => [c, h, w].into(),
                            dark_cfg::Shape::Dim1(c) => [c].into(),
                        },
                        DarkNodeKey::Index(_) => {
                            let from_keys = &input_keys_map[&key];
                            let layer_config = &layer_configs[&key];
                            // eprintln!("{}\t{:?}\t{}", key, from_keys, layer_config.as_ref());

                            let input_shape: dark_cfg::InputShape = match from_keys {
                                DarkInputKeys::Single(from_key) => {
                                    let input_shape = &collected[&from_key];
                                    let shape = input_shape.tensor().ok_or_else(|| {
                                        format_err!(
                                            "invalid input shape '{}' for layer {}",
                                            input_shape,
                                            key
                                        )
                                    })?;
                                    let shape: Vec<usize> = shape.try_into().unwrap();

                                    match *shape {
                                        [in_c] => [in_c].into(),
                                        [in_c, in_h, in_w] => [in_h, in_w, in_c].into(),
                                        _ => {
                                            bail!(
                                                "invalid input shape '{}' for layer {}",
                                                input_shape,
                                                key
                                            );
                                        }
                                    }
                                }
                                DarkInputKeys::Indexed(from_keys) => {
                                    let input_shapes: Vec<[usize; 3]> = from_keys
                                        .iter()
                                        .map(|&key| -> Result<_> {
                                            let shape = &collected[&key];
                                            let [in_c, in_h, in_w]: [usize; 3] =
                                                shape.tensor_nd().with_context(|| {
                                                    format!("invalid input shape from key: {}", key)
                                                })?;
                                            Ok([in_h, in_w, in_c])
                                        })
                                        .try_collect()?;
                                    input_shapes.into()
                                }
                                _ => unreachable!(),
                            };
                            let output_shape =
                                layer_config.output_shape(&input_shape).ok_or_else(|| {
                                    format_err!("cannot compute output shape at layer {}", key)
                                })?;

                            match output_shape {
                                dark_cfg::OutputShape::Shape(dark_cfg::Shape::Dim1(out_c)) => {
                                    [out_c].into()
                                }
                                dark_cfg::OutputShape::Shape(dark_cfg::Shape::Dim3(
                                    [out_h, out_w, out_c],
                                )) => [out_c, out_h, out_w].into(),
                                dark_cfg::OutputShape::Yolo(_shape) => ShapeOutput::Detect2D,
                            }
                        }
                    };

                    // eprintln!("{}\t{}", key, &output_shape);
                    collected.insert(key, output_shape);
                    Ok(collected)
                })()
                .with_context(|| format!("cannot compute output shape at layer {}", key))
            },
        )?;

        // aggregate all computed features
        let nodes: IndexMap<NodeKey, Node> = {
            let mut input_keys_map = input_keys_map;
            let mut layer_configs = layer_configs;
            let mut shapes_map = output_shape;

            sorted_node_keys
                .into_iter()
                .map(|key| -> Result<_> {
                    let input_keys = input_keys_map.remove(&key).unwrap();
                    let output_shape = shapes_map.remove(&key).unwrap();

                    let module: Module = match key {
                        DarkNodeKey::Input => {
                            let shape = match model_input_shape {
                                dark_cfg::Shape::Dim3([h, w, c]) => vec![c, h, w].into(),
                                dark_cfg::Shape::Dim1(size) => vec![size].into(),
                            };
                            Module::Input(config::Input {
                                name: "input".parse().unwrap(),
                                shape,
                            })
                        }
                        DarkNodeKey::Index(_) => {
                            let layer_config = layer_configs.remove(&key).unwrap().clone();

                            match layer_config {
                                dark_cfg::Layer::Convolutional(dark_cfg::Convolutional {
                                    filters,
                                    size,
                                    stride_x,
                                    stride_y,
                                    dilation,
                                    padding,
                                    groups,
                                    activation,
                                    batch_normalize,
                                    ..
                                }) => {
                                    let s = (stride_x == stride_y)
                                        .then(|| stride_x)
                                        .ok_or_else(|| format_err!("TODO"))?;

                                    config::ConvBn2D {
                                        name: None,
                                        from: None,
                                        c: filters,
                                        k: size,
                                        s,
                                        p: padding,
                                        d: dilation,
                                        g: groups,
                                        bias: true,
                                        act: activation.into(),
                                        bn: config::BatchNorm {
                                            enabled: batch_normalize,
                                            affine: true,
                                            var_min: None,
                                            var_max: None,
                                        },
                                    }
                                    .into()
                                }
                                dark_cfg::Layer::Connected(dark_cfg::Connected {
                                    output,
                                    batch_normalize,
                                    ..
                                }) => config::Linear {
                                    name: None,
                                    from: None,
                                    out: output,
                                    bn: config::BatchNorm {
                                        enabled: batch_normalize,
                                        affine: true,
                                        var_min: None,
                                        var_max: None,
                                    },
                                }
                                .into(),
                                dark_cfg::Layer::Route(dark_cfg::Route { group, .. }) => {
                                    config::DarknetRoute {
                                        name: None,
                                        from: None,
                                        group_id: group.group_id(),
                                        num_groups: group.num_groups(),
                                    }
                                    .into()
                                }
                                dark_cfg::Layer::Shortcut(dark_cfg::Shortcut {
                                    weights_type,
                                    ..
                                }) => config::DarknetShortcut {
                                    name: None,
                                    from: None,
                                    weights_type,
                                }
                                .into(),
                                dark_cfg::Layer::MaxPool(dark_cfg::MaxPool {
                                    stride_x,
                                    stride_y,
                                    size,
                                    padding,
                                    maxpool_depth,
                                    ..
                                }) => config::MaxPool {
                                    name: None,
                                    from: None,
                                    stride_x,
                                    stride_y,
                                    size,
                                    padding,
                                    maxpool_depth,
                                }
                                .into(),
                                dark_cfg::Layer::UpSample(dark_cfg::UpSample {
                                    stride,
                                    reverse,
                                    ..
                                }) => config::UpSample2D {
                                    name: None,
                                    from: None,
                                    config: config::UpSample2DConfig::ByStride { stride, reverse },
                                }
                                .into(),
                                dark_cfg::Layer::BatchNorm(dark_cfg::BatchNorm { .. }) => {
                                    todo!();
                                }
                                dark_cfg::Layer::Dropout(dark_cfg::Dropout { .. }) => {
                                    todo!();
                                }
                                dark_cfg::Layer::Softmax(dark_cfg::Softmax { .. }) => {
                                    todo!();
                                }
                                dark_cfg::Layer::Cost(dark_cfg::Cost { .. }) => {
                                    todo!();
                                }
                                dark_cfg::Layer::Crop(dark_cfg::Crop { .. }) => {
                                    todo!();
                                }
                                dark_cfg::Layer::AvgPool(dark_cfg::AvgPool { .. }) => {
                                    todo!();
                                }
                                dark_cfg::Layer::Yolo(dark_cfg::Yolo { .. }) => {
                                    todo!();
                                }
                                dark_cfg::Layer::GaussianYolo(dark_cfg::GaussianYolo {
                                    ..
                                }) => {
                                    todo!();
                                }
                                dark_cfg::Layer::Unimplemented(_) => {
                                    bail!("the layer {} is not implemented", key)
                                }
                            }
                        }
                    };

                    let node_key = NodeKey::from(key);
                    let input_keys: InputKeys = input_keys.into();
                    let node = Node {
                        input_keys,
                        output_shape,
                        path: None,
                        config: module,
                    };
                    Ok((node_key, node))
                })
                .try_collect()?
        };

        Ok(Graph { nodes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_graph_from_darknet_config_file() -> Result<()> {
        glob::glob(&format!(
            "{}/../darknet-config/cfg/yolov4-csp.cfg",
            env!("CARGO_MANIFEST_DIR")
        ))?
        .try_for_each(|file| -> Result<_> {
            let config = darknet_config::Darknet::load(file?)?;
            let _graph = Graph::from_darknet(&config)?;
            Ok(())
        })?;
        Ok(())
    }
}
