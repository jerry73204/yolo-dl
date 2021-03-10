use super::graph::*;
use crate::{
    common::*,
    config::{self, Module, Shape, ShapeOutput},
};
use darknet_config::config as dark_cfg;

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash)]
enum DarkNodeKey {
    Input,
    Index(usize),
}

impl DarkNodeKey {
    fn index(&self) -> Option<usize> {
        match *self {
            Self::Index(index) => Some(index),
            Self::Input => None,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum DarkInputKeys {
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
                                                || format_err!("invalid layer index"),
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
                                        .map(|&index| {
                                            let index = match index {
                                                dark_cfg::LayerIndex::Relative(index) => {
                                                    let index = index.get();
                                                    ensure!(
                                                        index <= layer_index,
                                                        "invalid layer index"
                                                    );
                                                    layer_index - index
                                                }
                                                dark_cfg::LayerIndex::Absolute(index) => index,
                                            };
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

            let sorted_node_keys = petgraph::algo::toposort(&graph, None).map_err(|cycle| {
                format_err!("cycle detected at layer index {:?}", cycle.node_id())
            })?;

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
                let output_shape = match key {
                    DarkNodeKey::Input => {
                        let shape: Vec<_> =
                            model_input_shape.iter().map(|size| size as usize).collect();
                        ShapeOutput::from(shape)
                    }
                    DarkNodeKey::Index(_) => {
                        let from_keys = &input_keys_map[&key];
                        let layer_config = &layer_configs[&key];

                        match layer_config {
                            dark_cfg::Layer::Convolutional(conf) => {
                                let from_key = from_keys.single().unwrap();
                                let [in_c, in_h, in_w]: [usize; 3] =
                                    collected[&from_key].tensor_nd().unwrap();
                                let [out_h, out_w, out_c] =
                                    conf.output_shape([in_h as u64, in_w as u64, in_c as u64]);
                                ShapeOutput::from([out_c as usize, out_h as usize, out_w as usize])
                            }
                            dark_cfg::Layer::Connected(conf) => {
                                ShapeOutput::from([conf.output as usize])
                            }
                            dark_cfg::Layer::Shortcut(_conf) => {
                                let input_shapes: Vec<[usize; 3]> = from_keys
                                    .indexed()
                                    .unwrap()
                                    .iter()
                                    .map(|&key| collected[&key].tensor_nd().unwrap())
                                    .collect();

                                // ensure input layers have equal heights and widths
                                ensure!(
                                    {
                                        let set: HashSet<_> =
                                            input_shapes.iter().map(|[_c, h, w]| [h, w]).collect();

                                        set.len() == 1
                                    },
                                    "the input layers must have equal heights and widths"
                                );

                                // copy the shape of first layer as output shape
                                let output_shape = input_shapes[0];

                                ShapeOutput::from(output_shape)
                            }
                            dark_cfg::Layer::MaxPool(conf) => {
                                let from_key = from_keys.single().unwrap();
                                let [in_c, in_h, in_w]: [usize; 3] =
                                    collected[&from_key].tensor_nd().unwrap();
                                let [out_h, out_w, out_c] =
                                    conf.output_shape([in_h as u64, in_w as u64, in_c as u64]);
                                ShapeOutput::from([out_c as usize, out_h as usize, out_w as usize])
                            }
                            dark_cfg::Layer::Route(conf) => {
                                let dark_cfg::Route { group, .. } = conf;
                                let num_groups = group.num_groups();

                                let input_shapes: Vec<[usize; 3]> = from_keys
                                    .indexed()
                                    .unwrap()
                                    .iter()
                                    .map(|&key| collected[&key].tensor_nd().unwrap())
                                    .collect();

                                ensure!(
                                    {
                                        let set: HashSet<_> =
                                            input_shapes.iter().map(|&[_c, h, w]| [h, w]).collect();
                                        set.len() == 1
                                    },
                                    "shape mismatch"
                                );

                                let [_c, out_h, out_w] = input_shapes[0];
                                let out_c: usize =
                                    input_shapes.iter().try_fold(0, |sum, &[in_c, _h, _w]| {
                                        ensure!(
                                            in_c % num_groups as usize == 0,
                                            "the input channel size must be multiple of groups"
                                        );
                                        Ok(sum + in_c / num_groups as usize)
                                    })?;
                                let output_shape = [out_c, out_h, out_w];
                                ShapeOutput::from(output_shape)
                            }
                            dark_cfg::Layer::UpSample(conf) => {
                                let from_key = from_keys.single().unwrap();
                                let [in_c, in_h, in_w]: [usize; 3] =
                                    collected[&from_key].tensor_nd().unwrap();
                                let [out_h, out_w, out_c] =
                                    conf.output_shape([in_h as u64, in_w as u64, in_c as u64]);
                                ShapeOutput::from([out_c as usize, out_h as usize, out_w as usize])
                            }
                            dark_cfg::Layer::Yolo(_) => ShapeOutput::Detect2D,
                            dark_cfg::Layer::GaussianYolo(_) => ShapeOutput::Detect2D,
                            dark_cfg::Layer::BatchNorm(_)
                            | dark_cfg::Layer::Dropout(_)
                            | dark_cfg::Layer::Softmax(_) => {
                                let from_key = from_keys.single().unwrap();
                                let input_shape: Shape = collected[&from_key].tensor_nd().unwrap();
                                ShapeOutput::from(input_shape)
                            }
                            _ => unimplemented!(),
                        }
                    }
                };

                collected.insert(key, output_shape);
                Ok(collected)
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
                        DarkNodeKey::Input => Module::Input(config::Input {
                            name: "input".parse().unwrap(),
                            shape: convert_darknet_shape(&model_input_shape),
                        }),
                        DarkNodeKey::Index(_) => {
                            let layer_config = layer_configs.remove(&key).unwrap().clone();

                            match layer_config {
                                dark_cfg::Layer::Convolutional(conf) => {
                                    let dark_cfg::Convolutional {
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
                                    } = conf;

                                    let s = (stride_x == stride_y)
                                        .then(|| stride_x as usize)
                                        .ok_or_else(|| format_err!("TODO"))?;

                                    Module::ConvBn2D(config::ConvBn2D {
                                        name: None,
                                        from: None,
                                        c: filters as usize,
                                        k: size as usize,
                                        s,
                                        p: padding as usize,
                                        d: dilation as usize,
                                        g: groups as usize,
                                        bias: true,
                                        act: activation.into(),
                                        bn: config::BatchNorm {
                                            enabled: batch_normalize,
                                            affine: true,
                                            var_min: None,
                                            var_max: None,
                                        },
                                    })
                                }
                                _ => unimplemented!(),
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

fn convert_darknet_shape(from: &dark_cfg::Shape) -> Shape {
    match *from {
        dark_cfg::Shape::Hwc([h, w, c]) => vec![c as usize, h as usize, w as usize].into(),
        dark_cfg::Shape::Flat(size) => vec![size as usize].into(),
    }
}
