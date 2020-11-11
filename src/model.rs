use crate::{
    common::*,
    config::{
        BatchNormConfig, CommonLayerOptions, ConnectedConfig, ConvolutionalConfig, DarknetConfig,
        LayerConfig, LayerIndex, MaxPoolConfig, NetConfig, RouteConfig, Shape, ShortcutConfig,
        UpSampleConfig, YoloConfig,
    },
    weights::{
        BatchNormWeights, ConnectedWeights, ConvolutionalWeights, ScaleWeights, ShortcutWeights,
    },
};

pub use layers::*;

#[derive(Debug)]
pub struct Model {
    seen: u64,
    cur_iteration: u64,
    net: NetConfig,
    layers: IndexMap<usize, Layer>,
}

impl Model {
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
                NetConfig {
                    input_size: model_input_shape,
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
                        let from_indexes_vec: Vec<_> = conf
                            .from
                            .iter()
                            .map(|index| -> Result<_> {
                                let index = index
                                    .to_absolute(layer_index)
                                    .ok_or_else(|| format_err!("invalid layer index"))?;
                                Ok(LayerPosition::Absolute(index))
                            })
                            .try_collect()?;

                        {
                            let mut set: IndexSet<_> = from_indexes_vec.iter().cloned().collect();
                            ensure!(
                                set.len() == from_indexes_vec.len(),
                                "from indexes must not be duplicated"
                            );

                            if layer_index == 0 {
                                let inserted = set.insert(LayerPosition::Input);
                                debug_assert!(inserted, "please report bug");
                            } else {
                                let inserted = set.insert(LayerPosition::Absolute(layer_index - 1));
                                ensure!(
                                    inserted,
                                    "previous layer index must not be contained in from option"
                                );
                            }

                            LayerPositionSet::Multiple(set)
                        }
                    }
                    LayerConfig::Route(conf) => {
                        let from_indexes_vec: Vec<_> = conf
                            .layers
                            .iter()
                            .map(|index| {
                                let index = match *index {
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
                        let from_indexes_set: IndexSet<_> =
                            from_indexes_vec.iter().cloned().collect();
                        ensure!(
                            from_indexes_set.len() == from_indexes_vec.len(),
                            "the layer indexes must not be duplicated"
                        );
                        LayerPositionSet::Multiple(from_indexes_set)
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
                            let output_shape = {
                                let mut iter = input_shapes.iter().collect::<HashSet<_>>().into_iter();
                                let first = iter.next().unwrap();
                                ensure!(
                                    matches!(iter.next(), None),
                                    "the shapes of layers connected to a shortcut layer must be equal, but found {:?}",
                                    input_shapes
                                );
                                first.to_owned()
                            };
                            (ShapeList::MultipleHwc(input_shapes), Shape::Hwc(output_shape))
                        },
                        LayerConfig::MaxPool(conf) => {
                            let input_shape = hwc_input_shape(from_index)
                                .ok_or_else(|| format_err!("invalid shape"))?;
                            let output_shape = conf.output_shape(input_shape);
                            (ShapeList::SingleHwc(input_shape), Shape::Hwc(output_shape))
                        }
                        LayerConfig::Route(_conf) => {
                            let input_shapes = multiple_hwc_input_shapes(from_index)
                                .ok_or_else(|| format_err!("invalid shape"))?;
                            let [out_h, out_w] = {
                                let set: HashSet<_> = input_shapes.iter().cloned().map(|[h, w, _c]| [h ,w]).collect();
                                ensure!(set.len() == 1, "output shapes of input layers to a route layer must have the same heights and widths");
                                set.into_iter().next().unwrap()
                            };
                            let out_c: u64 = input_shapes.iter().cloned().map(|[_h, _w, c]| c).sum();
                            let output_shape = [out_h, out_w, out_c];
                            (ShapeList::MultipleHwc(input_shapes), Shape::Hwc(output_shape))
                        }
                        LayerConfig::UpSample(conf) => {
                            let input_shape = hwc_input_shape(from_index)
                                .ok_or_else(|| format_err!("invalid shape"))?;
                            let output_shape = conf.output_shape(input_shape);
                            (ShapeList::SingleHwc(input_shape), Shape::Hwc(output_shape))
                        }
                        LayerConfig::Yolo(_) => {
                            let input_shape = hwc_input_shape(from_index)
                                .ok_or_else(|| format_err!("invalid shape"))?;
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
                            let weights = conf.build_weights(input_shape, output_shape);

                            Layer::Connected(ConnectedLayer {
                                config: conf,
                                weights,
                                from_indexes: from_indexes.single().unwrap(),
                                input_shape,
                                output_shape,
                            })
                        }
                        LayerConfig::Convolutional(conf) => {
                            let input_shape = input_shape.single_hwc().unwrap();
                            let output_shape = output_shape.hwc().unwrap();
                            let weights =
                                conf.build_weights(layer_index, input_shape, output_shape)?;

                            Layer::Convolutional(ConvolutionalLayer {
                                config: conf,
                                weights,
                                from_indexes: from_indexes.single().unwrap(),
                                input_shape,
                                output_shape,
                            })
                        }
                        LayerConfig::Route(conf) => Layer::Route(RouteLayer {
                            config: conf,
                            weights: (),
                            from_indexes: from_indexes.multiple().unwrap(),
                            input_shape: input_shape.multiple_hwc().unwrap(),
                            output_shape: output_shape.hwc().unwrap(),
                        }),
                        LayerConfig::Shortcut(conf) => {
                            let input_shape = input_shape.multiple_hwc().unwrap();
                            let output_shape = output_shape.hwc().unwrap();
                            let weights = conf.build_weights(&input_shape, output_shape);

                            Layer::Shortcut(ShortcutLayer {
                                config: conf,
                                weights,
                                from_indexes: from_indexes.multiple().unwrap(),
                                input_shape,
                                output_shape,
                            })
                        }
                        LayerConfig::MaxPool(conf) => Layer::MaxPool(MaxPoolLayer {
                            config: conf,
                            weights: (),
                            from_indexes: from_indexes.single().unwrap(),
                            input_shape: input_shape.single_hwc().unwrap(),
                            output_shape: output_shape.hwc().unwrap(),
                        }),
                        LayerConfig::UpSample(conf) => Layer::UpSample(UpSampleLayer {
                            config: conf,
                            weights: (),
                            from_indexes: from_indexes.single().unwrap(),
                            input_shape: input_shape.single_hwc().unwrap(),
                            output_shape: output_shape.hwc().unwrap(),
                        }),
                        LayerConfig::BatchNorm(conf) => {
                            let input_shape = input_shape.single_hwc().unwrap();
                            let output_shape = output_shape.hwc().unwrap();
                            let weights = conf.build_weights(input_shape, output_shape);

                            Layer::BatchNorm(BatchNormLayer {
                                config: conf,
                                weights,
                                from_indexes: from_indexes.single().unwrap(),
                                input_shape,
                                output_shape,
                            })
                        }
                        LayerConfig::Yolo(conf) => Layer::Yolo(YoloLayer {
                            config: conf,
                            weights: (),
                            from_indexes: from_indexes.single().unwrap(),
                            input_shape: input_shape.single_hwc().unwrap(),
                            output_shape: output_shape.hwc().unwrap(),
                        }),
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
                    Layer::Convolutional(_) => "conv",
                    Layer::Connected(_) => "connected",
                    Layer::BatchNorm(_) => "batch_norm",
                    Layer::Shortcut(_) => "shortcut",
                    Layer::MaxPool(_) => "max_pool",
                    Layer::Route(_) => "route",
                    Layer::UpSample(_) => "up_sample",
                    Layer::Yolo(_) => "yolo",
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

    pub fn load_weights<P>(&mut self, weights_file: P) -> Result<()>
    where
        P: AsRef<Path>,
    {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, BinRead)]
        pub struct Version {
            pub major: u32,
            pub minor: u32,
            pub revision: u32,
        }

        let mut reader = BufReader::new(File::open(weights_file)?);

        // load weights file
        let (seen, transpose, mut reader) = move || -> Result<_, binread::Error> {
            let version: Version = reader.read_le()?;
            let Version { major, minor, .. } = version;

            let seen: u64 = if major * 10 + minor >= 2 {
                reader.read_le()?
            } else {
                let seen: u32 = reader.read_le()?;
                seen as u64
            };
            let transpose = (major > 1000) || (minor > 1000);

            Ok((seen, transpose, reader))
        }()
        .map_err(|err| format_err!("failed to parse weight file: {:?}", err))?;

        // update network parameters
        self.seen = seen;
        self.cur_iteration = self.net.iteration(seen);

        // load weights
        {
            let num_layers = self.layers.len();

            (0..num_layers).try_for_each(|layer_index| -> Result<_> {
                let layer = &mut self.layers[&layer_index];
                layer.load_weights(&mut reader, transpose)?;
                Ok(())
            })?;

            ensure!(
                matches!(reader.fill_buf()?, &[]),
                "the weights file is not totally consumed"
            );
        }

        Ok(())
    }
}

mod layers {
    pub use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum LayerPosition {
        Input,
        Absolute(usize),
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

    #[derive(Debug)]
    pub enum Layer {
        Connected(ConnectedLayer),
        Convolutional(ConvolutionalLayer),
        Route(RouteLayer),
        Shortcut(ShortcutLayer),
        MaxPool(MaxPoolLayer),
        UpSample(UpSampleLayer),
        Yolo(YoloLayer),
        BatchNorm(BatchNormLayer),
    }

    impl Layer {
        pub fn load_weights(&mut self, reader: impl ReadBytesExt, transpose: bool) -> Result<()> {
            match self {
                Self::Connected(layer) => layer.load_weights(reader, transpose),
                Self::Convolutional(layer) => layer.load_weights(reader),
                Self::Route(_layer) => Ok(()),
                Self::Shortcut(layer) => layer.load_weights(reader),
                Self::MaxPool(_layer) => Ok(()),
                Self::UpSample(_layer) => Ok(()),
                Self::Yolo(_layer) => Ok(()),
                Self::BatchNorm(layer) => layer.load_weights(reader),
            }
        }

        pub fn input_shape(&self) -> ShapeList {
            match self {
                Self::Connected(layer) => ShapeList::SingleFlat(layer.input_shape),
                Self::Convolutional(layer) => ShapeList::SingleHwc(layer.input_shape),
                Self::Route(layer) => ShapeList::MultipleHwc(layer.input_shape.clone()),
                Self::Shortcut(layer) => ShapeList::MultipleHwc(layer.input_shape.clone()),
                Self::MaxPool(layer) => ShapeList::SingleHwc(layer.input_shape),
                Self::UpSample(layer) => ShapeList::SingleHwc(layer.input_shape),
                Self::Yolo(layer) => ShapeList::SingleHwc(layer.input_shape),
                Self::BatchNorm(layer) => ShapeList::SingleHwc(layer.input_shape),
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
                Self::Yolo(layer) => Shape::Hwc(layer.output_shape),
                Self::BatchNorm(layer) => Shape::Hwc(layer.output_shape),
            }
        }
    }

    macro_rules! declare_layer_type {
        ($name:ident, $config:ty, $weights:ty, $from_indexes:ty, $input_shape:ty, $output_shape:ty) => {
            #[derive(Debug)]
            pub struct $name {
                pub config: $config,
                pub weights: $weights,
                pub from_indexes: $from_indexes,
                pub input_shape: $input_shape,
                pub output_shape: $output_shape,
            }
        };
    }

    declare_layer_type!(
        ConnectedLayer,
        ConnectedConfig,
        ConnectedWeights,
        LayerPosition,
        u64,
        u64
    );
    declare_layer_type!(
        ConvolutionalLayer,
        ConvolutionalConfig,
        ConvolutionalWeights,
        LayerPosition,
        [u64; 3],
        [u64; 3]
    );
    declare_layer_type!(
        RouteLayer,
        RouteConfig,
        (),
        IndexSet<LayerPosition>,
        Vec<[u64; 3]>,
        [u64; 3]
    );
    declare_layer_type!(
        ShortcutLayer,
        ShortcutConfig,
        ShortcutWeights,
        IndexSet<LayerPosition>,
        Vec<[u64; 3]>,
        [u64; 3]
    );
    declare_layer_type!(
        MaxPoolLayer,
        MaxPoolConfig,
        (),
        LayerPosition,
        [u64; 3],
        [u64; 3]
    );
    declare_layer_type!(
        UpSampleLayer,
        UpSampleConfig,
        (),
        LayerPosition,
        [u64; 3],
        [u64; 3]
    );
    declare_layer_type!(YoloLayer, YoloConfig, (), LayerPosition, [u64; 3], [u64; 3]);
    declare_layer_type!(
        BatchNormLayer,
        BatchNormConfig,
        BatchNormWeights,
        LayerPosition,
        [u64; 3],
        [u64; 3]
    );

    impl From<ConnectedLayer> for Layer {
        fn from(from: ConnectedLayer) -> Self {
            Self::Connected(from)
        }
    }

    impl From<ConvolutionalLayer> for Layer {
        fn from(from: ConvolutionalLayer) -> Self {
            Self::Convolutional(from)
        }
    }

    impl From<RouteLayer> for Layer {
        fn from(from: RouteLayer) -> Self {
            Self::Route(from)
        }
    }

    impl From<ShortcutLayer> for Layer {
        fn from(from: ShortcutLayer) -> Self {
            Self::Shortcut(from)
        }
    }

    impl From<MaxPoolLayer> for Layer {
        fn from(from: MaxPoolLayer) -> Self {
            Self::MaxPool(from)
        }
    }

    impl From<UpSampleLayer> for Layer {
        fn from(from: UpSampleLayer) -> Self {
            Self::UpSample(from)
        }
    }

    impl From<YoloLayer> for Layer {
        fn from(from: YoloLayer) -> Self {
            Self::Yolo(from)
        }
    }

    impl From<BatchNormLayer> for Layer {
        fn from(from: BatchNormLayer) -> Self {
            Self::BatchNorm(from)
        }
    }

    impl ConnectedLayer {
        pub fn load_weights(
            &mut self,
            mut reader: impl ReadBytesExt,
            transpose: bool,
        ) -> Result<()> {
            let Self {
                config:
                    ConnectedConfig {
                        common:
                            CommonLayerOptions {
                                dont_load,
                                dont_load_scales,
                                ..
                            },
                        ..
                    },
                weights:
                    ConnectedWeights {
                        ref mut biases,
                        ref mut weights,
                        ref mut scales,
                    },
                input_shape,
                output_shape,
                ..
            } = *self;

            if dont_load {
                return Ok(());
            }

            reader.read_f32_into::<LittleEndian>(biases)?;
            reader.read_f32_into::<LittleEndian>(weights)?;

            if transpose {
                crate::utils::transpose_matrix(
                    weights,
                    input_shape as usize,
                    output_shape as usize,
                )?;
            }

            if let (Some(scales), false) = (scales, dont_load_scales) {
                let ScaleWeights {
                    scales,
                    rolling_mean,
                    rolling_variance,
                } = scales;

                reader.read_f32_into::<LittleEndian>(scales)?;
                reader.read_f32_into::<LittleEndian>(rolling_mean)?;
                reader.read_f32_into::<LittleEndian>(rolling_variance)?;
            }

            Ok(())
        }
    }

    impl ConvolutionalLayer {
        pub fn load_weights(&mut self, mut reader: impl ReadBytesExt) -> Result<()> {
            let Self {
                config:
                    ConvolutionalConfig {
                        groups,
                        size,
                        filters,
                        flipped,
                        common:
                            CommonLayerOptions {
                                dont_load,
                                dont_load_scales,
                                ..
                            },
                        ..
                    },
                ref mut weights,
                input_shape: [_h, _w, in_c],
                ..
            } = *self;

            if dont_load {
                return Ok(());
            }

            match weights {
                ConvolutionalWeights::Ref { .. } => (),
                ConvolutionalWeights::Owned {
                    biases,
                    scales,
                    weights,
                } => {
                    reader.read_f32_into::<LittleEndian>(biases)?;

                    if let (Some(scales), false) = (scales, dont_load_scales) {
                        let ScaleWeights {
                            scales,
                            rolling_mean,
                            rolling_variance,
                        } = scales;

                        reader.read_f32_into::<LittleEndian>(scales)?;
                        reader.read_f32_into::<LittleEndian>(rolling_mean)?;
                        reader.read_f32_into::<LittleEndian>(rolling_variance)?;
                    }

                    reader.read_f32_into::<LittleEndian>(weights)?;

                    if flipped {
                        crate::utils::transpose_matrix(
                            weights,
                            ((in_c / groups) * size.pow(2)) as usize,
                            filters as usize,
                        )?;
                    }
                }
            }

            Ok(())
        }
    }

    impl BatchNormLayer {
        pub fn load_weights(&mut self, mut reader: impl ReadBytesExt) -> Result<()> {
            let Self {
                config:
                    BatchNormConfig {
                        common: CommonLayerOptions { dont_load, .. },
                        ..
                    },
                weights:
                    BatchNormWeights {
                        ref mut biases,
                        ref mut scales,
                        ref mut rolling_mean,
                        ref mut rolling_variance,
                    },
                ..
            } = *self;

            if dont_load {
                return Ok(());
            }

            reader.read_f32_into::<LittleEndian>(biases)?;
            reader.read_f32_into::<LittleEndian>(scales)?;
            reader.read_f32_into::<LittleEndian>(rolling_mean)?;
            reader.read_f32_into::<LittleEndian>(rolling_variance)?;

            Ok(())
        }
    }

    impl ShortcutLayer {
        pub fn load_weights(&mut self, mut reader: impl ReadBytesExt) -> Result<()> {
            let Self {
                config:
                    ShortcutConfig {
                        common: CommonLayerOptions { dont_load, .. },
                        ..
                    },
                weights: ShortcutWeights { ref mut weights },
                ..
            } = *self;

            if dont_load {
                return Ok(());
            }

            if let Some(weights) = weights {
                reader.read_f32_into::<LittleEndian>(weights)?;
            }

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wtf() -> Result<()> {
        pretty_env_logger::init();
        let config_file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("yolov4.cfg");
        let weights_file = "/home/jerry73204/Downloads/yolov4.weights";
        let config = DarknetConfig::load(config_file)?;
        let mut model = Model::from_config(&config)?;
        model.load_weights(weights_file)?;
        Ok(())
    }
}
