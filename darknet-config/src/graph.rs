use crate::{
    common::*,
    config::{self, LayerIndex, Shape},
    utils::DisplayAsDebug,
};

pub use graph::*;
pub use node::*;
pub use node_key::*;
pub use shape::*;

mod graph {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct Graph {
        pub seen: u64,
        pub cur_iteration: u64,
        pub net: config::Net,
        pub layers: IndexMap<NodeKey, Node>,
    }

    impl Graph {
        pub fn from_config_file<P>(config_file: P) -> Result<Self>
        where
            P: AsRef<Path>,
        {
            let config = config::Darknet::load(config_file)?;
            let model = Self::from_config(&config)?;
            Ok(model)
        }

        pub fn from_config(config: &config::Darknet) -> Result<Self> {
            // load config file
            let config::Darknet {
                net:
                    config::Net {
                        input_size: model_input_shape,
                        ..
                    },
                ref layers,
            } = *config;

            // compute from indexes per layer
            let from_indexes_map: IndexMap<NodeKey, _> = layers
                .iter()
                .enumerate()
                .map(|(layer_index, layer_config)| -> Result<_> {
                    let from_indexes = match layer_config {
                        config::Layer::Convolutional(_)
                        | config::Layer::Connected(_)
                        | config::Layer::BatchNorm(_)
                        | config::Layer::MaxPool(_)
                        | config::Layer::UpSample(_)
                        | config::Layer::Dropout(_)
                        | config::Layer::Softmax(_)
                        | config::Layer::GaussianYolo(_)
                        | config::Layer::Yolo(_) => {
                            if layer_index == 0 {
                                InputKeys::Single(NodeKey::Input)
                            } else {
                                InputKeys::Single(NodeKey::Index(layer_index - 1))
                            }
                        }
                        config::Layer::Shortcut(conf) => {
                            let from = &conf.from;
                            let first_index = if layer_index == 0 {
                                NodeKey::Input
                            } else {
                                NodeKey::Index(layer_index - 1)
                            };

                            let from_indexes: IndexSet<_> = iter::once(Ok(first_index))
                                .chain(from.iter().map(|index| -> Result<_> {
                                    let index = index
                                        .to_absolute(layer_index)
                                        .ok_or_else(|| format_err!("invalid layer index"))?;
                                    Ok(NodeKey::Index(index))
                                }))
                                .try_collect()?;

                            ensure!(
                                from_indexes.len() == from.len() + 1,
                                "from must not contain the index to previous layer"
                            );

                            InputKeys::Multiple(from_indexes)
                        }
                        config::Layer::Route(conf) => {
                            let from_indexes: IndexSet<_> = conf
                                .layers
                                .iter()
                                .map(|&index| {
                                    let index = match index {
                                        LayerIndex::Relative(index) => {
                                            let index = index.get();
                                            ensure!(index <= layer_index, "invalid layer index");
                                            layer_index - index
                                        }
                                        LayerIndex::Absolute(index) => index,
                                    };
                                    Ok(NodeKey::Index(index))
                                })
                                .try_collect()?;
                            InputKeys::Multiple(from_indexes)
                        }
                        _ => unimplemented!(),
                    };

                    let key = NodeKey::Index(layer_index);
                    Ok((key, from_indexes))
                })
                .chain(iter::once(Ok((NodeKey::Input, InputKeys::None))))
                .try_collect()?;

            // topological sort
            let sorted_node_keys: Vec<NodeKey> = {
                let graph = {
                    let mut graph = DiGraphMap::<NodeKey, ()>::new();
                    from_indexes_map.iter().for_each(|(&key, from_indexes)| {
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

            let layer_configs_map: IndexMap<NodeKey, _> = sorted_node_keys
                .iter()
                .cloned()
                .filter_map(|key| Some((key, &layers[key.index()?])))
                .collect();

            // compute shapes
            let shapes_map: IndexMap<NodeKey, (ShapeList, Shape)> =
            sorted_node_keys.iter().try_fold(
                IndexMap::new(),
                |mut collected, &key| -> Result<_> {
                    // closures
                    let hwc_input_shape = |from_indexes: &InputKeys| {
                        let shape = match *from_indexes {
                            InputKeys::Single(src_key) => {
                                let (_input_shape, output_shape) =
                                    collected.get(&src_key).expect("please report bug");
                                *output_shape
                            }
                            _ => return None,
                        };
                        match shape {
                            Shape::Hwc(hwc) => Some(hwc),
                            Shape::Flat(_) => None,
                        }
                    };
                    let flat_input_shape = |from_indexes: &InputKeys| {
                        let shape = match *from_indexes {
                            InputKeys::Single(src_key) => {
                                let ( _input_shape, output_shape) =
                                    collected.get(&src_key).expect("please report bug");
                                *output_shape
                            }
                            _ => return None,
                        };
                        match shape {
                            Shape::Flat(flat) => Some(flat),
                            Shape::Hwc(_) => None,
                        }
                    };
                    let multiple_hwc_input_shapes = |from_index: &InputKeys| match from_index {
                        InputKeys::Multiple(src_keys) => {
                            src_keys
                                .iter()
                                .cloned()
                                .map(|src_key| {
                                    let shape = {
                                        let (_input_shape, output_shape) =
                                            collected.get(&src_key).expect("please report bug");
                                        *output_shape
                                    };
                                    match shape {
                                        Shape::Hwc(hwc) => Some(hwc),
                                        Shape::Flat(_) => None,
                                    }
                                })
                                .try_fold(vec![], |mut folded, shape| {
                                    let shape = shape?;
                                    folded.push(shape);
                                    Some(folded)
                                })
                        }
                        _ => None,
                    };

                    let (input_shape, output_shape) = match key {
                        NodeKey::Input => {
                            (ShapeList::None, model_input_shape)
                        }
                        _ => {
                            let from_index = from_indexes_map.get(&key).expect("please report bug");
                            let layer_config = layer_configs_map.get(&key).expect("please report bug");

                            let (input_shape, output_shape) = match layer_config {
                                config::Layer::Convolutional(conf) => {
                                    let input_shape = hwc_input_shape(from_index)
                                        .ok_or_else(|| format_err!("invalid shape"))?;
                                    let output_shape = conf.output_shape(input_shape);
                                    (ShapeList::SingleHwc(input_shape), Shape::Hwc(output_shape))
                                }
                                config::Layer::Connected(conf) => {
                                    let input_shape = flat_input_shape(from_index)
                                        .ok_or_else(|| format_err!("invalid shape"))?;
                                    let output_shape = conf.output;
                                    (ShapeList::SingleFlat(input_shape), Shape::Flat(output_shape))
                                }
                                config::Layer::BatchNorm(_conf) => {
                                    let input_shape = hwc_input_shape(from_index)
                                        .ok_or_else(|| format_err!("invalid shape"))?;
                                    let output_shape = input_shape;
                                    (ShapeList::SingleHwc(input_shape), Shape::Hwc(output_shape))
                                }
                                config::Layer::Shortcut(_conf) => {
                                    let input_shapes = multiple_hwc_input_shapes(from_index)
                                        .ok_or_else(|| format_err!("invalid shape"))?;

                                    // ensure input layers have equal heights and widths
                                    {
                                        let set: HashSet<_> = input_shapes.iter().map(|[h, w, _c]| [h, w]).collect();
                                        ensure!(set.len() == 1, "the input layers must have equal heights and widths");
                                    }

                                    // copy the shape of first layer as output shape
                                    let output_shape = input_shapes[0];

                                    (ShapeList::MultipleHwc(input_shapes), Shape::Hwc(output_shape))
                                },
                                config::Layer::MaxPool(conf) => {
                                    let input_shape = hwc_input_shape(from_index)
                                        .ok_or_else(|| format_err!("invalid shape"))?;
                                    let output_shape = conf.output_shape(input_shape);
                                    (ShapeList::SingleHwc(input_shape), Shape::Hwc(output_shape))
                                }
                                config::Layer::Dropout(_conf) => {
                                    let input_shape = hwc_input_shape(from_index)
                                        .ok_or_else(|| format_err!("invalid shape"))?;
                                    let output_shape = input_shape;
                                    (ShapeList::SingleHwc(input_shape), Shape::Hwc(output_shape))
                                }
                                config::Layer::Softmax(_conf) => {
                                    let input_shape = hwc_input_shape(from_index)
                                        .ok_or_else(|| format_err!("invalid shape"))?;
                                    let output_shape = input_shape;
                                    (ShapeList::SingleHwc(input_shape), Shape::Hwc(output_shape))
                                }
                                config::Layer::Route(conf) => {
                                    let config::Route { group, .. } = conf;

                                    let num_groups = group.num_groups();

                                    let input_shapes = multiple_hwc_input_shapes(from_index)
                                        .ok_or_else(|| format_err!("invalid shape"))?;
                                    let [out_h, out_w] = {
                                        let set: HashSet<_> = input_shapes.iter().cloned().map(|[h, w, _c]| [h ,w]).collect();
                                        ensure!(set.len() == 1, "output shapes of input layers to a route layer must have the same heights and widths");
                                        set.into_iter().next().unwrap()
                                    };
                                    let out_c: u64 = input_shapes.iter().cloned().try_fold(0, |sum, [_h, _w, c]| {
                                        ensure!(c % num_groups == 0, "the input channel size must be multiple of groups");
                                        Ok(sum + c / num_groups)
                                    })?;
                                    let output_shape = [out_h, out_w, out_c];
                                    (ShapeList::MultipleHwc(input_shapes), Shape::Hwc(output_shape))
                                }
                                config::Layer::UpSample(conf) => {
                                    let input_shape = hwc_input_shape(from_index)
                                        .ok_or_else(|| format_err!("invalid shape"))?;
                                    let output_shape = conf.output_shape(input_shape);
                                    (ShapeList::SingleHwc(input_shape), Shape::Hwc(output_shape))
                                }
                                config::Layer::Yolo(conf) => {
                                    let [in_h, in_w, in_c] = hwc_input_shape(from_index)
                                        .ok_or_else(|| format_err!("invalid shape"))?;
                                    let config::Yolo {
                                        classes, anchors, ..
                                    } = conf;

                                    // [batch, anchor, entry, h, w]
                                    let num_anchors = anchors.len() as u64;
                                    ensure!(in_c == num_anchors * (classes + 4 + 1), "the output channels and yolo input channels mismatch");

                                    let input_shape = [in_h, in_w, in_c];
                                    let output_shape = input_shape;
                                    (ShapeList::SingleHwc(input_shape), Shape::Hwc(output_shape))
                                }
                                config::Layer::GaussianYolo(conf) => {
                                    let [in_h, in_w, in_c] = hwc_input_shape(from_index)
                                        .ok_or_else(|| format_err!("invalid shape"))?;
                                    let config::GaussianYolo {
                                        classes, anchors, ..
                                    } = conf;

                                    // [batch, anchor, entry, h, w]
                                    let num_anchors = anchors.len() as u64;
                                    ensure!(in_c == num_anchors * (classes + 4 + 1), "the output channels and yolo input channels mismatch");

                                    let input_shape = [in_h, in_w, in_c];
                                    let output_shape = input_shape;
                                    (ShapeList::SingleHwc(input_shape), Shape::Hwc(output_shape))
                                }
                                _ => unimplemented!(),
                            };

                            (input_shape, output_shape)
                        }
                    };


                    collected.insert(key, (input_shape, output_shape));

                    Ok(collected)
                },
            )?;

            // aggregate all computed features
            let layers: IndexMap<_, _> = {
                let mut from_indexes_map = from_indexes_map;
                let mut layer_configs_map = layer_configs_map;
                let mut shapes_map = shapes_map;

                sorted_node_keys
                    .into_iter()
                    .map(|key| -> Result<_> {
                        let node = match key {
                            NodeKey::Input => {
                                let (_input_shape, output_shape) = shapes_map.remove(&key).unwrap();
                                Node::Input(InputNode { output_shape })
                            }
                            _ => {
                                let from_indexes = from_indexes_map.remove(&key).unwrap();
                                let (input_shape, output_shape) = shapes_map.remove(&key).unwrap();
                                let layer_config = layer_configs_map.remove(&key).unwrap().clone();

                                let node = match layer_config {
                                    config::Layer::Connected(conf) => {
                                        let input_shape = input_shape.single_flat().unwrap();
                                        let output_shape = output_shape.flat().unwrap();

                                        Node::Connected(ConnectedNode {
                                            config: conf,
                                            from_indexes: from_indexes.single().unwrap(),
                                            input_shape,
                                            output_shape,
                                        })
                                    }
                                    config::Layer::Convolutional(conf) => {
                                        let input_shape = input_shape.single_hwc().unwrap();
                                        let output_shape = output_shape.hwc().unwrap();

                                        Node::Convolutional(ConvolutionalNode {
                                            config: conf,
                                            from_indexes: from_indexes.single().unwrap(),
                                            input_shape,
                                            output_shape,
                                        })
                                    }
                                    config::Layer::Route(conf) => {
                                        let input_shape = input_shape.multiple_hwc().unwrap();
                                        let output_shape = output_shape.hwc().unwrap();

                                        Node::Route(RouteNode {
                                            config: conf,
                                            from_indexes: from_indexes.multiple().unwrap(),
                                            input_shape,
                                            output_shape,
                                        })
                                    }
                                    config::Layer::Shortcut(conf) => {
                                        let input_shape = input_shape.multiple_hwc().unwrap();
                                        let output_shape = output_shape.hwc().unwrap();

                                        Node::Shortcut(ShortcutNode {
                                            config: conf,
                                            from_indexes: from_indexes.multiple().unwrap(),
                                            input_shape,
                                            output_shape,
                                        })
                                    }
                                    config::Layer::MaxPool(conf) => {
                                        let input_shape = input_shape.single_hwc().unwrap();
                                        let output_shape = output_shape.hwc().unwrap();
                                        Node::MaxPool(MaxPoolNode {
                                            config: conf,
                                            from_indexes: from_indexes.single().unwrap(),
                                            input_shape,
                                            output_shape,
                                        })
                                    }
                                    config::Layer::UpSample(conf) => {
                                        let input_shape = input_shape.single_hwc().unwrap();
                                        let output_shape = output_shape.hwc().unwrap();

                                        Node::UpSample(UpSampleNode {
                                            config: conf,
                                            from_indexes: from_indexes.single().unwrap(),
                                            input_shape,
                                            output_shape,
                                        })
                                    }
                                    config::Layer::BatchNorm(conf) => {
                                        let input_shape = input_shape.single_hwc().unwrap();
                                        let output_shape = output_shape.hwc().unwrap();
                                        debug_assert_eq!(input_shape, output_shape);

                                        Node::BatchNorm(BatchNormNode {
                                            config: conf,
                                            from_indexes: from_indexes.single().unwrap(),
                                            inout_shape: input_shape,
                                        })
                                    }
                                    config::Layer::Dropout(conf) => {
                                        let input_shape = input_shape.single_hwc().unwrap();
                                        let output_shape = output_shape.hwc().unwrap();
                                        debug_assert_eq!(input_shape, output_shape);

                                        Node::Dropout(DropoutNode {
                                            config: conf,
                                            from_indexes: from_indexes.single().unwrap(),
                                            inout_shape: input_shape,
                                        })
                                    }
                                    config::Layer::Softmax(conf) => {
                                        let input_shape = input_shape.single_hwc().unwrap();
                                        let output_shape = output_shape.hwc().unwrap();
                                        debug_assert_eq!(input_shape, output_shape);

                                        Node::Softmax(SoftmaxNode {
                                            config: conf,
                                            from_indexes: from_indexes.single().unwrap(),
                                            inout_shape: input_shape,
                                        })
                                    }
                                    config::Layer::Yolo(conf) => {
                                        let input_shape = input_shape.single_hwc().unwrap();
                                        let output_shape = output_shape.hwc().unwrap();
                                        debug_assert_eq!(input_shape, output_shape);

                                        Node::Yolo(YoloNode {
                                            config: conf,
                                            from_indexes: from_indexes.single().unwrap(),
                                            inout_shape: input_shape,
                                        })
                                    }
                                    config::Layer::GaussianYolo(conf) => {
                                        let input_shape = input_shape.single_hwc().unwrap();
                                        let output_shape = output_shape.hwc().unwrap();
                                        debug_assert_eq!(input_shape, output_shape);

                                        Node::GaussianYolo(GaussianYoloNode {
                                            config: conf,
                                            from_indexes: from_indexes.single().unwrap(),
                                            inout_shape: input_shape,
                                        })
                                    }
                                    _ => unimplemented!(),
                                };

                                node
                            }
                        };

                        Ok((key, node))
                    })
                    .try_collect()?
            };

            // network parameters
            let net = config.net.clone();
            let seen = 0;
            let cur_iteration = 0;

            Ok(Self {
                seen,
                cur_iteration,
                net,
                layers,
            })
        }
    }
}

// node key

mod node_key {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum NodeKey {
        Input,
        Index(usize),
    }

    impl NodeKey {
        pub fn index(&self) -> Option<usize> {
            match *self {
                Self::Input => None,
                Self::Index(index) => Some(index),
            }
        }
    }

    impl Display for NodeKey {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Input => write!(f, "input"),
                Self::Index(index) => write!(f, "{}", index),
            }
        }
    }

    impl PartialOrd for NodeKey {
        fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
            match (self, rhs) {
                (Self::Input, Self::Input) => Some(Ordering::Equal),
                (Self::Input, Self::Index(_)) => Some(Ordering::Less),
                (Self::Index(_), Self::Input) => Some(Ordering::Greater),
                (Self::Index(lindex), Self::Index(rindex)) => lindex.partial_cmp(rindex),
            }
        }
    }

    impl Ord for NodeKey {
        fn cmp(&self, rhs: &Self) -> Ordering {
            match (self, rhs) {
                (Self::Input, Self::Input) => Ordering::Equal,
                (Self::Input, Self::Index(_)) => Ordering::Less,
                (Self::Index(_), Self::Input) => Ordering::Greater,
                (Self::Index(lindex), Self::Index(rindex)) => lindex.cmp(rindex),
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum InputKeys {
        None,
        Single(NodeKey),
        Multiple(IndexSet<NodeKey>),
    }

    impl InputKeys {
        pub fn iter(&self) -> impl Iterator<Item = NodeKey> {
            let index_iter: Box<dyn Iterator<Item = NodeKey>> = match *self {
                Self::None => Box::new(iter::empty()),
                Self::Single(index) => Box::new(iter::once(index)),
                Self::Multiple(ref indexes) => Box::new(indexes.clone().into_iter()),
            };
            index_iter
        }

        pub fn single(&self) -> Option<NodeKey> {
            match *self {
                Self::Single(index) => Some(index),
                _ => None,
            }
        }

        pub fn multiple(&self) -> Option<IndexSet<NodeKey>> {
            match self {
                Self::Multiple(indexes) => Some(indexes.clone()),
                _ => None,
            }
        }
    }

    impl Display for InputKeys {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::None => write!(f, "none"),
                Self::Single(index) => write!(f, "{}", index),
                Self::Multiple(indexes) => f
                    .debug_list()
                    .entries(indexes.iter().cloned().map(|index| DisplayAsDebug(index)))
                    .finish(),
            }
        }
    }
}

// shape

mod shape {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum ShapeList {
        None,
        SingleFlat(u64),
        SingleHwc([u64; 3]),
        MultipleHwc(Vec<[u64; 3]>),
    }

    impl ShapeList {
        pub fn is_none(&self) -> bool {
            match self {
                Self::None => true,
                _ => false,
            }
        }

        pub fn single_flat(&self) -> Option<u64> {
            match *self {
                Self::SingleFlat(size) => Some(size),
                _ => None,
            }
        }

        pub fn single_hwc(&self) -> Option<[u64; 3]> {
            match *self {
                Self::SingleHwc(hwc) => Some(hwc),
                _ => None,
            }
        }

        pub fn multiple_hwc(&self) -> Option<Vec<[u64; 3]>> {
            match self {
                Self::MultipleHwc(hwc) => Some(hwc.clone()),
                _ => None,
            }
        }
    }

    impl Display for ShapeList {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::None => write!(f, "[none]"),
                Self::SingleFlat(size) => write!(f, "[{}]", size),
                Self::SingleHwc([h, w, c]) => f.debug_list().entries(vec![h, w, c]).finish(),
                Self::MultipleHwc(shapes) => write!(f, "{:?}", shapes),
            }
        }
    }
}

// layer

mod node {
    use super::*;

    #[derive(Debug, Clone, AsRefStr)]
    pub enum Node {
        Input(InputNode),
        Connected(ConnectedNode),
        Convolutional(ConvolutionalNode),
        Route(RouteNode),
        Shortcut(ShortcutNode),
        MaxPool(MaxPoolNode),
        UpSample(UpSampleNode),
        BatchNorm(BatchNormNode),
        Dropout(DropoutNode),
        Softmax(SoftmaxNode),
        Yolo(YoloNode),
        GaussianYolo(GaussianYoloNode),
    }

    impl Node {
        pub fn input_shape(&self) -> ShapeList {
            match self {
                Self::Input(_layer) => ShapeList::None,
                Self::Connected(layer) => ShapeList::SingleFlat(layer.input_shape),
                Self::Convolutional(layer) => ShapeList::SingleHwc(layer.input_shape),
                Self::Route(layer) => ShapeList::MultipleHwc(layer.input_shape.clone()),
                Self::Shortcut(layer) => ShapeList::MultipleHwc(layer.input_shape.clone()),
                Self::MaxPool(layer) => ShapeList::SingleHwc(layer.input_shape),
                Self::UpSample(layer) => ShapeList::SingleHwc(layer.input_shape),
                Self::BatchNorm(layer) => ShapeList::SingleHwc(layer.inout_shape),
                Self::Dropout(layer) => ShapeList::SingleHwc(layer.inout_shape),
                Self::Softmax(layer) => ShapeList::SingleHwc(layer.inout_shape),
                Self::Yolo(layer) => ShapeList::SingleHwc(layer.inout_shape),
                Self::GaussianYolo(layer) => ShapeList::SingleHwc(layer.inout_shape),
            }
        }

        pub fn output_shape(&self) -> Shape {
            match self {
                Self::Input(layer) => layer.output_shape.to_owned(),
                Self::Connected(layer) => Shape::Flat(layer.output_shape),
                Self::Convolutional(layer) => Shape::Hwc(layer.output_shape),
                Self::Route(layer) => Shape::Hwc(layer.output_shape),
                Self::Shortcut(layer) => Shape::Hwc(layer.output_shape),
                Self::MaxPool(layer) => Shape::Hwc(layer.output_shape),
                Self::UpSample(layer) => Shape::Hwc(layer.output_shape),
                Self::BatchNorm(layer) => Shape::Hwc(layer.inout_shape),
                Self::Dropout(layer) => Shape::Hwc(layer.inout_shape),
                Self::Softmax(layer) => Shape::Hwc(layer.inout_shape),
                Self::Yolo(layer) => Shape::Hwc(layer.inout_shape),
                Self::GaussianYolo(layer) => Shape::Hwc(layer.inout_shape),
            }
        }

        pub fn from_indexes(&self) -> InputKeys {
            match self {
                Self::Input(_layer) => InputKeys::None,
                Self::Connected(layer) => InputKeys::Single(layer.from_indexes),
                Self::Convolutional(layer) => InputKeys::Single(layer.from_indexes),
                Self::Route(layer) => InputKeys::Multiple(layer.from_indexes.clone()),
                Self::Shortcut(layer) => InputKeys::Multiple(layer.from_indexes.clone()),
                Self::MaxPool(layer) => InputKeys::Single(layer.from_indexes),
                Self::UpSample(layer) => InputKeys::Single(layer.from_indexes),
                Self::BatchNorm(layer) => InputKeys::Single(layer.from_indexes),
                Self::Dropout(layer) => InputKeys::Single(layer.from_indexes),
                Self::Softmax(layer) => InputKeys::Single(layer.from_indexes),
                Self::Yolo(layer) => InputKeys::Single(layer.from_indexes),
                Self::GaussianYolo(layer) => InputKeys::Single(layer.from_indexes),
            }
        }
    }

    macro_rules! declare_layer_base_inout_shape {
        ($name:ident, $config:ty, $from_indexes:ty, $input_shape:ty, $output_shape:ty) => {
            #[derive(Debug, Clone)]
            pub struct $name {
                pub config: $config,
                pub from_indexes: $from_indexes,
                pub input_shape: $input_shape,
                pub output_shape: $output_shape,
            }
        };
    }

    macro_rules! declare_layer_base_single_shape {
        ($name:ident, $config:ty, $from_indexes:ty, $inout_shape:ty) => {
            #[derive(Debug, Clone)]
            pub struct $name {
                pub config: $config,
                pub from_indexes: $from_indexes,
                pub inout_shape: $inout_shape,
            }
        };
    }

    #[derive(Debug, Clone)]
    pub struct InputNode {
        pub output_shape: Shape,
    }

    declare_layer_base_inout_shape!(ConnectedNode, config::Connected, NodeKey, u64, u64);
    declare_layer_base_inout_shape!(
        ConvolutionalNode,
        config::Convolutional,
        NodeKey,
        [u64; 3],
        [u64; 3]
    );
    declare_layer_base_inout_shape!(
        RouteNode,
        config::Route,
        IndexSet<NodeKey>,
        Vec<[u64; 3]>,
        [u64; 3]
    );
    declare_layer_base_inout_shape!(
        ShortcutNode,
        config::Shortcut,
        IndexSet<NodeKey>,
        Vec<[u64; 3]>,
        [u64; 3]
    );
    declare_layer_base_inout_shape!(MaxPoolNode, config::MaxPool, NodeKey, [u64; 3], [u64; 3]);
    declare_layer_base_inout_shape!(UpSampleNode, config::UpSample, NodeKey, [u64; 3], [u64; 3]);
    declare_layer_base_single_shape!(YoloNode, config::Yolo, NodeKey, [u64; 3]);
    declare_layer_base_single_shape!(GaussianYoloNode, config::GaussianYolo, NodeKey, [u64; 3]);
    declare_layer_base_single_shape!(BatchNormNode, config::BatchNorm, NodeKey, [u64; 3]);
    declare_layer_base_single_shape!(DropoutNode, config::Dropout, NodeKey, [u64; 3]);
    declare_layer_base_single_shape!(SoftmaxNode, config::Softmax, NodeKey, [u64; 3]);

    impl From<ConnectedNode> for Node {
        fn from(from: ConnectedNode) -> Self {
            Self::Connected(from)
        }
    }

    impl From<ConvolutionalNode> for Node {
        fn from(from: ConvolutionalNode) -> Self {
            Self::Convolutional(from)
        }
    }

    impl From<RouteNode> for Node {
        fn from(from: RouteNode) -> Self {
            Self::Route(from)
        }
    }

    impl From<ShortcutNode> for Node {
        fn from(from: ShortcutNode) -> Self {
            Self::Shortcut(from)
        }
    }

    impl From<MaxPoolNode> for Node {
        fn from(from: MaxPoolNode) -> Self {
            Self::MaxPool(from)
        }
    }

    impl From<UpSampleNode> for Node {
        fn from(from: UpSampleNode) -> Self {
            Self::UpSample(from)
        }
    }

    impl From<YoloNode> for Node {
        fn from(from: YoloNode) -> Self {
            Self::Yolo(from)
        }
    }

    impl From<BatchNormNode> for Node {
        fn from(from: BatchNormNode) -> Self {
            Self::BatchNorm(from)
        }
    }
}

// graphviz support

#[cfg(feature = "dot")]
mod graphviz {
    use super::*;
    use dot::{Arrow, Edges, GraphWalk, Id, LabelText, Labeller, Nodes, Style};

    impl Graph {
        pub fn render_dot(&self, writer: &mut impl Write) -> Result<()> {
            dot::render(self, writer)?;
            Ok(())
        }
    }

    // `None` for input node, Some(index) for layer at that index

    impl<'a> GraphWalk<'a, NodeKey, (NodeKey, NodeKey)> for Graph {
        fn nodes(&'a self) -> Nodes<'a, NodeKey> {
            self.layers.keys().cloned().collect_vec().into()
        }

        fn edges(&'a self) -> Edges<'a, (NodeKey, NodeKey)> {
            self.layers
                .iter()
                .flat_map(|(&key, layer)| {
                    layer
                        .from_indexes()
                        .iter()
                        .map(move |from_index| (from_index, key))
                })
                .collect_vec()
                .into()
        }

        fn source(&'a self, edge: &(NodeKey, NodeKey)) -> NodeKey {
            let (src, _dst) = *edge;
            src
        }

        fn target(&'a self, edge: &(NodeKey, NodeKey)) -> NodeKey {
            let (_src, dst) = *edge;
            dst
        }
    }

    impl<'a> Labeller<'a, NodeKey, (NodeKey, NodeKey)> for Graph {
        fn graph_id(&'a self) -> Id<'a> {
            Id::new("darknet").unwrap()
        }

        fn node_id(&'a self, &key: &NodeKey) -> Id<'a> {
            match key {
                NodeKey::Input => Id::new("input").unwrap(),
                NodeKey::Index(layer_index) => Id::new(format!("layer_{}", layer_index)).unwrap(),
            }
        }

        fn node_shape(&'a self, &key: &NodeKey) -> Option<LabelText<'a>> {
            match key {
                NodeKey::Input => Some(LabelText::label("box")),
                _ => match self.layers[&key] {
                    Node::Yolo(_) | Node::GaussianYolo(_) => Some(LabelText::label("box")),
                    Node::Shortcut(_) => Some(LabelText::label("invtrapezium")),
                    Node::Route(_) => Some(LabelText::label("invhouse")),
                    _ => None,
                },
            }
        }

        fn node_label(&'a self, &key: &NodeKey) -> LabelText<'a> {
            match key {
                NodeKey::Input => LabelText::escaped(format!(
                    r"input
{}",
                    dot::escape_html(&format!("{:?}", self.net.input_size))
                )),
                NodeKey::Index(layer_index) => {
                    let output_shape = self.layers[&key].output_shape();

                    match &self.layers[&key] {
                        Node::Convolutional(node) => {
                            let ConvolutionalNode {
                                config:
                                    config::Convolutional {
                                        size,
                                        stride_x,
                                        stride_y,
                                        dilation,
                                        padding,
                                        share_index,
                                        batch_normalize,
                                        ..
                                    },
                                ..
                            } = *node;

                            match share_index {
                                Some(share_index) => LabelText::escaped(format!(
                                    r"({}) {}
{}
share={}",
                                    layer_index,
                                    self.layers[&key].as_ref(),
                                    dot::escape_html(&format!("{:?}", output_shape)),
                                    share_index.absolute().unwrap()
                                )),
                                None => {
                                    if stride_y == stride_x {
                                        LabelText::escaped(format!(
                                            r"({}) {}
{}
k={} s={} p={} d={}
batch_norm={}",
                                            layer_index,
                                            self.layers[&key].as_ref(),
                                            dot::escape_html(&format!("{:?}", output_shape)),
                                            size,
                                            stride_y,
                                            padding,
                                            dilation,
                                            if batch_normalize { "yes" } else { "no" }
                                        ))
                                    } else {
                                        LabelText::escaped(format!(
                                            r"({}) {}
{}
k={} sy={} sx={} p={} d={}
batch_norm={}",
                                            layer_index,
                                            self.layers[&key].as_ref(),
                                            dot::escape_html(&format!("{:?}", output_shape)),
                                            size,
                                            stride_y,
                                            stride_x,
                                            padding,
                                            dilation,
                                            if batch_normalize { "yes" } else { "no" }
                                        ))
                                    }
                                }
                            }
                        }
                        Node::MaxPool(node) => {
                            let MaxPoolNode {
                                config:
                                    config::MaxPool {
                                        size,
                                        stride_x,
                                        stride_y,
                                        padding,
                                        ..
                                    },
                                ..
                            } = *node;

                            if stride_y == stride_x {
                                LabelText::escaped(format!(
                                    r"({}) {}
{}
k={} s={} p={}",
                                    layer_index,
                                    self.layers[&key].as_ref(),
                                    dot::escape_html(&format!("{:?}", output_shape)),
                                    size,
                                    stride_y,
                                    padding,
                                ))
                            } else {
                                LabelText::escaped(format!(
                                    r"({}) {}
{}
k={} sy={} sx={} p={}",
                                    layer_index,
                                    self.layers[&key].as_ref(),
                                    dot::escape_html(&format!("{:?}", output_shape)),
                                    size,
                                    stride_y,
                                    stride_x,
                                    padding,
                                ))
                            }
                        }
                        _ => LabelText::escaped(format!(
                            r"({}) {}
{}",
                            layer_index,
                            self.layers[&key].as_ref(),
                            dot::escape_html(&format!("{:?}", output_shape))
                        )),
                    }
                }
            }
        }

        fn node_style(&'a self, _node: &NodeKey) -> Style {
            Style::None
        }

        fn node_color(&'a self, &key: &NodeKey) -> Option<LabelText<'a>> {
            match key {
                NodeKey::Input => Some(LabelText::label("black")),
                _ => match self.layers[&key] {
                    Node::Yolo(_) | Node::GaussianYolo(_) => Some(LabelText::label("orange")),
                    Node::Convolutional(_) => Some(LabelText::label("blue")),
                    Node::MaxPool(_) => Some(LabelText::label("green")),
                    Node::Shortcut(_) => Some(LabelText::label("brown")),
                    Node::Route(_) => Some(LabelText::label("brown")),
                    _ => None,
                },
            }
        }

        fn edge_label(&'a self, &edge: &(NodeKey, NodeKey)) -> LabelText<'a> {
            match edge {
                (NodeKey::Input, NodeKey::Index(to_index)) => {
                    LabelText::label(format!("input -> {}", to_index))
                }
                (NodeKey::Index(from_index), NodeKey::Index(to_index)) => {
                    LabelText::escaped(format!(
                        r"{} -> {}
{}",
                        from_index,
                        to_index,
                        dot::escape_html(&format!("{:?}", self.layers[from_index].output_shape()))
                    ))
                }
                _ => unreachable!(),
            }
        }

        fn edge_start_arrow(&'a self, _edge: &(NodeKey, NodeKey)) -> Arrow {
            Arrow::none()
        }

        fn edge_end_arrow(&'a self, _edge: &(NodeKey, NodeKey)) -> Arrow {
            Arrow::normal()
        }

        fn edge_style(&'a self, _node: &(NodeKey, NodeKey)) -> Style {
            Style::None
        }

        fn edge_color(&'a self, _node: &(NodeKey, NodeKey)) -> Option<LabelText<'a>> {
            None
        }

        fn kind(&self) -> dot::Kind {
            dot::Kind::Digraph
        }
    }
}
