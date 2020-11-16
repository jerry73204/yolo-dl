use crate::{
    common::*,
    config::{
        BatchNormConfig, CompoundNetConfig, CompoundYoloConfig, ConnectedConfig,
        ConvolutionalConfig, DarknetConfig, LayerConfig, LayerIndex, MaxPoolConfig, RouteConfig,
        Shape, ShortcutConfig, UpSampleConfig,
    },
    utils::DisplayAsDebug,
};

#[derive(Debug, Clone)]
pub struct ModelBase {
    pub seen: u64,
    pub cur_iteration: u64,
    pub net: CompoundNetConfig,
    pub layers: IndexMap<usize, LayerBase>,
}

impl ModelBase {
    pub fn from_config_file<P>(config_file: P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let config = DarknetConfig::load(config_file)?;
        let model = Self::from_config(&config)?;
        Ok(model)
    }

    pub fn from_config(config: &DarknetConfig) -> Result<Self> {
        // load config file
        let DarknetConfig {
            net:
                CompoundNetConfig {
                    input_size: model_input_shape,
                    classes: num_classes,
                    ..
                },
            ref layers,
        } = *config;

        // compute from indexes per layer
        let from_indexes_map: IndexMap<_, _> = layers
            .iter()
            .enumerate()
            .map(|(layer_index, layer_config)| -> Result<_> {
                let from_indexes = match layer_config {
                    LayerConfig::Convolutional(_)
                    | LayerConfig::Connected(_)
                    | LayerConfig::BatchNorm(_)
                    | LayerConfig::MaxPool(_)
                    | LayerConfig::UpSample(_)
                    | LayerConfig::Yolo(_) => {
                        if layer_index == 0 {
                            LayerPositionSet::Single(LayerPosition::Input)
                        } else {
                            LayerPositionSet::Single(LayerPosition::Absolute(layer_index - 1))
                        }
                    }
                    LayerConfig::Shortcut(conf) => {
                        let from = &conf.from;
                        let first_index = if layer_index == 0 {
                            LayerPosition::Input
                        } else {
                            LayerPosition::Absolute(layer_index - 1)
                        };

                        let from_indexes: IndexSet<_> = iter::once(Ok(first_index))
                            .chain(from.iter().map(|index| -> Result<_> {
                                let index = index
                                    .to_absolute(layer_index)
                                    .ok_or_else(|| format_err!("invalid layer index"))?;
                                Ok(LayerPosition::Absolute(index))
                            }))
                            .try_collect()?;

                        ensure!(
                            from_indexes.len() == from.len() + 1,
                            "from must not contain the index to previous layer"
                        );

                        LayerPositionSet::Multiple(from_indexes)
                    }
                    LayerConfig::Route(conf) => {
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
                                Ok(LayerPosition::Absolute(index))
                            })
                            .try_collect()?;
                        LayerPositionSet::Multiple(from_indexes)
                    }
                };
                Ok((layer_index, from_indexes))
            })
            .try_collect()?;

        // topological sort
        let sorted_layer_indexes = {
            let graph = {
                let mut graph = DiGraphMap::<LayerPosition, ()>::new();
                graph.add_node(LayerPosition::Input);
                from_indexes_map
                    .iter()
                    .for_each(|(&layer_index, from_indexes)| {
                        graph.add_node(LayerPosition::Absolute(layer_index));
                        from_indexes.iter().for_each(|from_index| {
                            graph.add_edge(from_index, LayerPosition::Absolute(layer_index), ());
                        });
                    });
                graph
            };

            let sorted_layer_indexes: Vec<_> = {
                let mut sorted_iter = petgraph::algo::toposort(&graph, None)
                    .map_err(|cycle| {
                        format_err!("cycle detected at layer index {:?}", cycle.node_id())
                    })?
                    .into_iter();

                let first = sorted_iter.next();
                debug_assert!(
                    matches!(first, Some(LayerPosition::Input)),
                    "please report bug"
                );

                sorted_iter
                    .map(|layer_index| match layer_index {
                        LayerPosition::Input => unreachable!("please report bug"),
                        LayerPosition::Absolute(index) => index,
                    })
                    .collect()
            };

            sorted_layer_indexes
        };

        let layer_configs_map: IndexMap<_, _> = sorted_layer_indexes
            .iter()
            .cloned()
            .map(|layer_index| (layer_index, &layers[layer_index]))
            .collect();

        // compute shapes
        let shapes_map: IndexMap<usize, (ShapeList, Shape)> =
            sorted_layer_indexes.iter().try_fold(
                IndexMap::new(),
                |mut collected, layer_index| -> Result<_> {
                    // closures
                    let hwc_input_shape = |from_indexes: &LayerPositionSet| {
                        let shape = match *from_indexes {
                            LayerPositionSet::Single(LayerPosition::Input) => {
                                model_input_shape
                            }
                            LayerPositionSet::Single(LayerPosition::Absolute(index)) => {
                                let (_input_shape, output_shape) =
                                    collected.get(&index).expect("please report bug");
                                *output_shape
                            }
                            _ => return None,
                        };
                        match shape {
                            Shape::Hwc(hwc) => Some(hwc),
                            Shape::Flat(_) => None,
                        }
                    };
                    let flat_input_shape = |from_indexes: &LayerPositionSet| {
                        let shape = match *from_indexes {
                            LayerPositionSet::Single(LayerPosition::Input) => {
                                model_input_shape
                            }
                            LayerPositionSet::Single(LayerPosition::Absolute(index)) => {
                                let ( _input_shape, output_shape) =
                                    collected.get(&index).expect("please report bug");
                                *output_shape
                            }
                            _ => return None,
                        };
                        match shape {
                            Shape::Flat(flat) => Some(flat),
                            Shape::Hwc(_) => None,
                        }
                    };
                    let multiple_hwc_input_shapes = |from_index: &LayerPositionSet| match from_index {
                        LayerPositionSet::Multiple(indexes) => {
                            indexes
                                .iter()
                                .cloned()
                                .map(|index| {
                                    let shape = match index {
                                        LayerPosition::Input => {
                                            model_input_shape
                                        },
                                        LayerPosition::Absolute(index) => {
                                            let (_input_shape, output_shape) =
                                                collected.get(&index).expect("please report bug");
                                            *output_shape
                                        },
                                    };
                                    match shape {
                                        Shape::Hwc(hwc) => Some(hwc),
                                        Shape::Flat(_) => None,
                                    }
                                }).fold(Some(vec![]), |folded, shape| {
                                    match (folded, shape) {
                                        (Some(mut folded), Some(shape)) => {
                                            folded.push(shape);
                                            Some(folded)
                                        }
                                        _ => None
                                    }
                                })
                        }
                        _ => None,
                    };

                    let from_index = from_indexes_map.get(layer_index).expect("please report bug");
                    let layer_config = layer_configs_map.get(layer_index).expect("please report bug");

                    let (input_shape, output_shape) = match layer_config {
                        LayerConfig::Convolutional(conf) => {
                            let input_shape = hwc_input_shape(from_index)
                                .ok_or_else(|| format_err!("invalid shape"))?;
                            let output_shape = conf.output_shape(input_shape);
                            (ShapeList::SingleHwc(input_shape), Shape::Hwc(output_shape))
                        }
                        LayerConfig::Connected(conf) => {
                            let input_shape = flat_input_shape(from_index)
                                .ok_or_else(|| format_err!("invalid shape"))?;
                            let output_shape = conf.output;
                            (ShapeList::SingleFlat(input_shape), Shape::Flat(output_shape))
                        }
                        LayerConfig::BatchNorm(_conf) => {
                            let input_shape = hwc_input_shape(from_index)
                                .ok_or_else(|| format_err!("invalid shape"))?;
                            let output_shape = input_shape;
                            (ShapeList::SingleHwc(input_shape), Shape::Hwc(output_shape))
                        }
                        LayerConfig::Shortcut(_conf) => {
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
                        LayerConfig::MaxPool(conf) => {
                            let input_shape = hwc_input_shape(from_index)
                                .ok_or_else(|| format_err!("invalid shape"))?;
                            let output_shape = conf.output_shape(input_shape);
                            (ShapeList::SingleHwc(input_shape), Shape::Hwc(output_shape))
                        }
                        LayerConfig::Route(conf) => {
                            let RouteConfig { group, .. } = conf;

                            let group_index = group.group_id();
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
                        LayerConfig::UpSample(conf) => {
                            let input_shape = hwc_input_shape(from_index)
                                .ok_or_else(|| format_err!("invalid shape"))?;
                            let output_shape = conf.output_shape(input_shape);
                            (ShapeList::SingleHwc(input_shape), Shape::Hwc(output_shape))
                        }
                        LayerConfig::Yolo(conf) => {
                            let [in_h, in_w, in_c] = hwc_input_shape(from_index)
                                .ok_or_else(|| format_err!("invalid shape"))?;
                            let CompoundYoloConfig {
                                anchors, ..
                            } = conf;

                            // [batch, anchor, entry, h, w]
                            let num_anchors = anchors.len() as u64;
                            ensure!(in_c == num_anchors * (num_classes + 4 + 1), "the output channels and yolo input channels mismatch");

                            let input_shape = [in_h, in_w, in_c];
                            let output_shape = input_shape;
                            (ShapeList::SingleHwc(input_shape), Shape::Hwc(output_shape))
                        }
                    };

                    collected.insert(*layer_index, (input_shape, output_shape));

                    Ok(collected)
                },
            )?;

        // aggregate all computed features
        let layers: IndexMap<_, _> = {
            let mut from_indexes_map = from_indexes_map;
            let mut layer_configs_map = layer_configs_map;
            let mut shapes_map = shapes_map;

            sorted_layer_indexes
                .into_iter()
                .map(|layer_index| -> Result<_> {
                    let from_indexes = from_indexes_map.remove(&layer_index).unwrap();
                    let (input_shape, output_shape) = shapes_map.remove(&layer_index).unwrap();
                    let layer_config = layer_configs_map.remove(&layer_index).unwrap().clone();

                    let layer = match layer_config {
                        LayerConfig::Connected(conf) => {
                            let input_shape = input_shape.single_flat().unwrap();
                            let output_shape = output_shape.flat().unwrap();

                            LayerBase::Connected(ConnectedLayerBase {
                                config: conf,
                                from_indexes: from_indexes.single().unwrap(),
                                input_shape,
                                output_shape,
                            })
                        }
                        LayerConfig::Convolutional(conf) => {
                            let input_shape = input_shape.single_hwc().unwrap();
                            let output_shape = output_shape.hwc().unwrap();

                            LayerBase::Convolutional(ConvolutionalLayerBase {
                                config: conf,
                                from_indexes: from_indexes.single().unwrap(),
                                input_shape,
                                output_shape,
                            })
                        }
                        LayerConfig::Route(conf) => {
                            let input_shape = input_shape.multiple_hwc().unwrap();
                            let output_shape = output_shape.hwc().unwrap();

                            LayerBase::Route(RouteLayerBase {
                                config: conf,
                                from_indexes: from_indexes.multiple().unwrap(),
                                input_shape,
                                output_shape,
                            })
                        }
                        LayerConfig::Shortcut(conf) => {
                            let input_shape = input_shape.multiple_hwc().unwrap();
                            let output_shape = output_shape.hwc().unwrap();

                            LayerBase::Shortcut(ShortcutLayerBase {
                                config: conf,
                                from_indexes: from_indexes.multiple().unwrap(),
                                input_shape,
                                output_shape,
                            })
                        }
                        LayerConfig::MaxPool(conf) => {
                            let input_shape = input_shape.single_hwc().unwrap();
                            let output_shape = output_shape.hwc().unwrap();
                            LayerBase::MaxPool(MaxPoolLayerBase {
                                config: conf,
                                from_indexes: from_indexes.single().unwrap(),
                                input_shape,
                                output_shape,
                            })
                        }
                        LayerConfig::UpSample(conf) => {
                            let input_shape = input_shape.single_hwc().unwrap();
                            let output_shape = output_shape.hwc().unwrap();

                            LayerBase::UpSample(UpSampleLayerBase {
                                config: conf,
                                from_indexes: from_indexes.single().unwrap(),
                                input_shape,
                                output_shape,
                            })
                        }
                        LayerConfig::BatchNorm(conf) => {
                            let input_shape = input_shape.single_hwc().unwrap();
                            let output_shape = output_shape.hwc().unwrap();
                            debug_assert_eq!(input_shape, output_shape);

                            LayerBase::BatchNorm(BatchNormLayerBase {
                                config: conf,
                                from_indexes: from_indexes.single().unwrap(),
                                inout_shape: input_shape,
                            })
                        }
                        LayerConfig::Yolo(conf) => {
                            let input_shape = input_shape.single_hwc().unwrap();
                            let output_shape = output_shape.hwc().unwrap();
                            debug_assert_eq!(input_shape, output_shape);

                            LayerBase::Yolo(YoloLayerBase {
                                config: conf,
                                from_indexes: from_indexes.single().unwrap(),
                                inout_shape: input_shape,
                            })
                        }
                    };

                    Ok((layer_index, layer))
                })
                .try_collect()?
        };

        // network parameters
        let net = config.net.clone();
        let seen = 0;
        let cur_iteration = 0;

        // print layer params for debugging
        #[cfg(debug_assertions)]
        {
            let num_layers = layers.len();
            (0..num_layers).for_each(|layer_index| {
                let layer = &layers[&layer_index];
                let kind = match layer {
                    LayerBase::Convolutional(_) => "conv",
                    LayerBase::Connected(_) => "connected",
                    LayerBase::BatchNorm(_) => "batch_norm",
                    LayerBase::Shortcut(_) => "shortcut",
                    LayerBase::MaxPool(_) => "max_pool",
                    LayerBase::Route(_) => "route",
                    LayerBase::UpSample(_) => "up_sample",
                    LayerBase::Yolo(_) => "yolo",
                };

                debug!(
                    "{}\t{}\t{:?}\t{:?}",
                    layer_index,
                    kind,
                    layer.input_shape(),
                    layer.output_shape()
                );
            });
        }

        Ok(Self {
            seen,
            cur_iteration,
            net,
            layers,
        })
    }
}

// layer position

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerPosition {
    Input,
    Absolute(usize),
}

impl Display for LayerPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input => write!(f, "input"),
            Self::Absolute(index) => write!(f, "{}", index),
        }
    }
}

impl PartialOrd for LayerPosition {
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        match (self, rhs) {
            (Self::Input, Self::Input) => Some(Ordering::Equal),
            (Self::Input, Self::Absolute(_)) => Some(Ordering::Less),
            (Self::Absolute(_), Self::Input) => Some(Ordering::Greater),
            (Self::Absolute(lindex), Self::Absolute(rindex)) => lindex.partial_cmp(rindex),
        }
    }
}

impl Ord for LayerPosition {
    fn cmp(&self, rhs: &Self) -> Ordering {
        match (self, rhs) {
            (Self::Input, Self::Input) => Ordering::Equal,
            (Self::Input, Self::Absolute(_)) => Ordering::Less,
            (Self::Absolute(_), Self::Input) => Ordering::Greater,
            (Self::Absolute(lindex), Self::Absolute(rindex)) => lindex.cmp(rindex),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerPositionSet {
    Empty,
    Single(LayerPosition),
    Multiple(IndexSet<LayerPosition>),
}

impl LayerPositionSet {
    pub fn iter(&self) -> impl Iterator<Item = LayerPosition> {
        let index_iter: Box<dyn Iterator<Item = LayerPosition>> = match *self {
            Self::Empty => Box::new(iter::empty()),
            Self::Single(index) => Box::new(iter::once(index)),
            Self::Multiple(ref indexes) => Box::new(indexes.clone().into_iter()),
        };
        index_iter
    }

    pub fn single(&self) -> Option<LayerPosition> {
        match *self {
            Self::Single(index) => Some(index),
            _ => None,
        }
    }

    pub fn multiple(&self) -> Option<IndexSet<LayerPosition>> {
        match self {
            Self::Multiple(indexes) => Some(indexes.clone()),
            _ => None,
        }
    }
}

impl Display for LayerPositionSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "empty"),
            Self::Single(index) => write!(f, "{}", index),
            Self::Multiple(indexes) => f
                .debug_list()
                .entries(indexes.iter().cloned().map(|index| DisplayAsDebug(index)))
                .finish(),
        }
    }
}

// shape

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeList {
    SingleFlat(u64),
    SingleHwc([u64; 3]),
    MultipleHwc(Vec<[u64; 3]>),
}

impl ShapeList {
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
            Self::SingleFlat(size) => write!(f, "{}", size),
            Self::SingleHwc([h, w, c]) => f.debug_list().entries(vec![h, w, c]).finish(),
            Self::MultipleHwc(shapes) => write!(f, "{:?}", shapes),
        }
    }
}

// layer

#[derive(Debug, Clone)]
pub enum LayerBase {
    Connected(ConnectedLayerBase),
    Convolutional(ConvolutionalLayerBase),
    Route(RouteLayerBase),
    Shortcut(ShortcutLayerBase),
    MaxPool(MaxPoolLayerBase),
    UpSample(UpSampleLayerBase),
    Yolo(YoloLayerBase),
    BatchNorm(BatchNormLayerBase),
}

impl LayerBase {
    pub fn input_shape(&self) -> ShapeList {
        match self {
            Self::Connected(layer) => ShapeList::SingleFlat(layer.input_shape),
            Self::Convolutional(layer) => ShapeList::SingleHwc(layer.input_shape),
            Self::Route(layer) => ShapeList::MultipleHwc(layer.input_shape.clone()),
            Self::Shortcut(layer) => ShapeList::MultipleHwc(layer.input_shape.clone()),
            Self::MaxPool(layer) => ShapeList::SingleHwc(layer.input_shape),
            Self::UpSample(layer) => ShapeList::SingleHwc(layer.input_shape),
            Self::Yolo(layer) => ShapeList::SingleHwc(layer.inout_shape),
            Self::BatchNorm(layer) => ShapeList::SingleHwc(layer.inout_shape),
        }
    }

    pub fn output_shape(&self) -> Shape {
        match self {
            Self::Connected(layer) => Shape::Flat(layer.output_shape),
            Self::Convolutional(layer) => Shape::Hwc(layer.output_shape),
            Self::Route(layer) => Shape::Hwc(layer.output_shape),
            Self::Shortcut(layer) => Shape::Hwc(layer.output_shape),
            Self::MaxPool(layer) => Shape::Hwc(layer.output_shape),
            Self::UpSample(layer) => Shape::Hwc(layer.output_shape),
            Self::Yolo(layer) => Shape::Hwc(layer.inout_shape),
            Self::BatchNorm(layer) => Shape::Hwc(layer.inout_shape),
        }
    }

    pub fn from_indexes(&self) -> LayerPositionSet {
        match self {
            Self::Connected(layer) => LayerPositionSet::Single(layer.from_indexes),
            Self::Convolutional(layer) => LayerPositionSet::Single(layer.from_indexes),
            Self::Route(layer) => LayerPositionSet::Multiple(layer.from_indexes.clone()),
            Self::Shortcut(layer) => LayerPositionSet::Multiple(layer.from_indexes.clone()),
            Self::MaxPool(layer) => LayerPositionSet::Single(layer.from_indexes),
            Self::UpSample(layer) => LayerPositionSet::Single(layer.from_indexes),
            Self::Yolo(layer) => LayerPositionSet::Single(layer.from_indexes),
            Self::BatchNorm(layer) => LayerPositionSet::Single(layer.from_indexes),
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

declare_layer_base_inout_shape!(ConnectedLayerBase, ConnectedConfig, LayerPosition, u64, u64);
declare_layer_base_inout_shape!(
    ConvolutionalLayerBase,
    ConvolutionalConfig,
    LayerPosition,
    [u64; 3],
    [u64; 3]
);
declare_layer_base_inout_shape!(
    RouteLayerBase,
    RouteConfig,
    IndexSet<LayerPosition>,
    Vec<[u64; 3]>,
    [u64; 3]
);
declare_layer_base_inout_shape!(
    ShortcutLayerBase,
    ShortcutConfig,
    IndexSet<LayerPosition>,
    Vec<[u64; 3]>,
    [u64; 3]
);
declare_layer_base_inout_shape!(
    MaxPoolLayerBase,
    MaxPoolConfig,
    LayerPosition,
    [u64; 3],
    [u64; 3]
);
declare_layer_base_inout_shape!(
    UpSampleLayerBase,
    UpSampleConfig,
    LayerPosition,
    [u64; 3],
    [u64; 3]
);
declare_layer_base_single_shape!(YoloLayerBase, CompoundYoloConfig, LayerPosition, [u64; 3]);
declare_layer_base_single_shape!(BatchNormLayerBase, BatchNormConfig, LayerPosition, [u64; 3]);

impl From<ConnectedLayerBase> for LayerBase {
    fn from(from: ConnectedLayerBase) -> Self {
        Self::Connected(from)
    }
}

impl From<ConvolutionalLayerBase> for LayerBase {
    fn from(from: ConvolutionalLayerBase) -> Self {
        Self::Convolutional(from)
    }
}

impl From<RouteLayerBase> for LayerBase {
    fn from(from: RouteLayerBase) -> Self {
        Self::Route(from)
    }
}

impl From<ShortcutLayerBase> for LayerBase {
    fn from(from: ShortcutLayerBase) -> Self {
        Self::Shortcut(from)
    }
}

impl From<MaxPoolLayerBase> for LayerBase {
    fn from(from: MaxPoolLayerBase) -> Self {
        Self::MaxPool(from)
    }
}

impl From<UpSampleLayerBase> for LayerBase {
    fn from(from: UpSampleLayerBase) -> Self {
        Self::UpSample(from)
    }
}

impl From<YoloLayerBase> for LayerBase {
    fn from(from: YoloLayerBase) -> Self {
        Self::Yolo(from)
    }
}

impl From<BatchNormLayerBase> for LayerBase {
    fn from(from: BatchNormLayerBase) -> Self {
        Self::BatchNorm(from)
    }
}

impl ConvolutionalLayerBase {
    pub fn weights_shape(&self) -> [u64; 4] {
        let Self {
            config:
                ConvolutionalConfig {
                    groups,
                    filters,
                    size,
                    ..
                },
            input_shape: [_h, _w, in_c],
            ..
        } = *self;

        debug_assert!(in_c % groups == 0,);
        [in_c / groups, filters, size, size]
    }
}
