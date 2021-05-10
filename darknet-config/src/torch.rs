use crate::{
    common::*,
    config::{self, Shape, WeightsNormalization, WeightsType},
    darknet::{self, DarknetModel},
    graph::{
        BatchNormNode, ConnectedNode, ConvolutionalNode, GaussianYoloNode, Graph, InputKeys,
        InputNode, MaxPoolNode, Node, NodeKey, RouteNode, ShapeList, ShortcutNode, UpSampleNode,
        YoloNode,
    },
};
use tch::{nn, IndexOp, Kind, Tensor};

pub use layer::*;
pub use tch_model::*;
pub use weights::*;

trait ReplaceTensor {
    fn replace(&mut self, data: &[f32], expect_shape: &[i64]);
}

impl ReplaceTensor for Tensor {
    fn replace(&mut self, data: &[f32], expect_shape: &[i64]) {
        tch::no_grad(|| {
            debug_assert_eq!(self.size(), expect_shape, "please report bug");
            debug_assert_eq!(self.kind(), Kind::Float);
            let new = Tensor::of_slice(data)
                .view(expect_shape)
                .to_device(self.device());
            let _ = mem::replace(self, new);
        });
    }
}

#[derive(Debug)]
pub enum TensorList {
    Single(Tensor),
    Multiple(Vec<Tensor>),
}

impl TensorList {
    pub fn shallow_clone(&self) -> Self {
        match self {
            Self::Single(tensor) => Self::Single(tensor.shallow_clone()),
            Self::Multiple(tensors) => Self::Multiple(
                tensors
                    .iter()
                    .map(|tensor| tensor.shallow_clone())
                    .collect(),
            ),
        }
    }

    pub fn single(&self) -> Option<&Tensor> {
        match self {
            Self::Single(tensor) => Some(tensor),
            _ => None,
        }
    }

    pub fn multiple(&self) -> Option<&[Tensor]> {
        match self {
            Self::Multiple(tensors) => Some(tensors),
            _ => None,
        }
    }
}

mod tch_model {
    use super::*;

    #[derive(Debug)]
    pub struct TchModel {
        pub graph: Graph,
        pub layers: IndexMap<NodeKey, Layer>,
    }

    impl TchModel {
        pub fn from_config_file<'p, P>(
            path: impl Borrow<nn::Path<'p>>,
            config_file: P,
        ) -> Result<Self>
        where
            P: AsRef<Path>,
        {
            let model = DarknetModel::from_config_file(config_file)?;
            Self::from_darknet_model(path, &model)
        }

        pub fn from_config<'p>(
            path: impl Borrow<nn::Path<'p>>,
            config: &config::Darknet,
        ) -> Result<Self> {
            let graph = Graph::from_config(config)?;
            Self::from_graph(path, &graph)
        }

        pub fn from_graph<'p>(path: impl Borrow<nn::Path<'p>>, graph: &Graph) -> Result<Self> {
            let path = path.borrow();

            let layers: IndexMap<NodeKey, _> = graph.layers.iter().try_fold(
                IndexMap::new(),
                |mut collected, (&key, layer_node)| -> Result<_> {
                    let layer: Layer = match layer_node {
                        Node::Input(node) => InputLayer::from_node(path, node)?.into(),
                        Node::Connected(node) => ConnectedLayer::from_node(path, node)?.into(),
                        Node::Convolutional(node) => ConvolutionalLayer::from_node(
                            path,
                            node,
                            key.index().unwrap(),
                            &collected,
                        )?
                        .into(),
                        Node::BatchNorm(node) => BatchNormLayer::from_node(path, node)?.into(),
                        Node::MaxPool(node) => MaxPoolLayer::from_node(path, node)?.into(),
                        Node::UpSample(node) => UpSampleLayer::from_node(path, node)?.into(),
                        Node::Shortcut(node) => ShortcutLayer::from_node(path, node)?.into(),
                        Node::Route(node) => RouteLayer::from_node(path, node)?.into(),
                        Node::Yolo(node) => YoloLayer::from_node(path, node)?.into(),
                        Node::GaussianYolo(node) => {
                            GaussianYoloLayer::from_node(path, node)?.into()
                        }
                        _ => unimplemented!(),
                    };

                    collected.insert(key, layer);
                    Ok(collected)
                },
            )?;

            Ok(Self {
                graph: graph.clone(),
                layers,
            })
        }

        pub fn from_darknet_model<'p>(
            path: impl Borrow<nn::Path<'p>>,
            model: &DarknetModel,
        ) -> Result<Self> {
            let path = path.borrow();

            let layers: IndexMap<NodeKey, _> = model.layers.iter().try_fold(
                IndexMap::new(),
                |mut collected, (&key, layer)| -> Result<_> {
                    let layer: Layer = match layer {
                        darknet::Layer::Input(conf) => InputLayer::from_darknet(path, conf)?.into(),
                        darknet::Layer::Connected(conf) => {
                            ConnectedLayer::from_darknet(path, conf)?.into()
                        }
                        darknet::Layer::Convolutional(conf) => {
                            ConvolutionalLayer::from_darknet(path, conf, &collected)?.into()
                        }
                        darknet::Layer::BatchNorm(conf) => {
                            BatchNormLayer::from_darknet(path, conf)?.into()
                        }
                        darknet::Layer::MaxPool(conf) => {
                            MaxPoolLayer::from_darknet(path, conf)?.into()
                        }
                        darknet::Layer::UpSample(conf) => {
                            UpSampleLayer::from_darknet(path, conf)?.into()
                        }
                        darknet::Layer::Shortcut(conf) => {
                            ShortcutLayer::from_darknet(path, conf)?.into()
                        }
                        darknet::Layer::Route(conf) => RouteLayer::from_darknet(path, conf)?.into(),
                        darknet::Layer::Yolo(conf) => YoloLayer::from_darknet(path, conf)?.into(),
                        darknet::Layer::GaussianYolo(conf) => {
                            GaussianYoloLayer::from_darknet(path, conf)?.into()
                        }
                    };

                    collected.insert(key, layer);
                    Ok(collected)
                },
            )?;

            Ok(Self {
                graph: model.graph.clone(),
                layers,
            })
        }

        pub fn input_shape(&self) -> ShapeList {
            // locate input layers
            let shape = match &self.layers[&NodeKey::Input] {
                Layer::Input(layer) => layer.node.output_shape,
                _ => unreachable!(),
            };

            match shape {
                Shape::Dim1(size) => ShapeList::SingleFlat(size),
                Shape::Dim3(size) => ShapeList::SingleHwc(size),
            }
        }

        pub fn forward_t(
            &mut self,
            input: &Tensor,
            train: bool,
        ) -> Result<(MultiDenseDetection, Vec<LayerOutput>)> {
            let (_batch_size, _channels, input_h, input_w) = input.size4()?;

            let mut feature_maps: HashMap<NodeKey, LayerOutput> = hashmap! {};
            let mut detections: HashMap<NodeKey, _> = hashmap! {};

            self.layers
                .iter_mut()
                .try_for_each(|(&key, layer)| -> Result<_> {
                    // get input tensors

                    let input = match key {
                        NodeKey::Input => TensorList::Single(input.shallow_clone()),
                        NodeKey::Index(_) => match layer.from_indexes() {
                            InputKeys::Single(src_key) => {
                                let tensor =
                                    feature_maps[&src_key].shallow_clone().tensor().unwrap();
                                TensorList::Single(tensor)
                            }
                            InputKeys::Multiple(src_keys) => {
                                let tensors: Vec<_> = src_keys
                                    .iter()
                                    .map(|src_key| {
                                        feature_maps[&src_key].shallow_clone().tensor().unwrap()
                                    })
                                    .collect();
                                TensorList::Multiple(tensors)
                            }
                            InputKeys::None => unreachable!(),
                        },
                    };

                    // save output
                    let output = layer.forward_t(input, train)?;
                    let prev = feature_maps.insert(key, output.shallow_clone());
                    debug_assert!(matches!(prev, None));

                    if let LayerOutput::Yolo(output) = output {
                        let prev = detections.insert(key, output);
                        debug_assert!(matches!(prev, None));
                    }

                    Ok(())
                })?;

            // pack output tensors
            let detections_iter = {
                let mut detections: Vec<_> = detections.into_iter().collect();
                detections.sort_by_cached_key(|(layer_index, _output)| *layer_index);
                detections.into_iter().map(|(_layer_index, output)| output)
            };

            // convert feature maps
            let feature_maps: Vec<_> = {
                let mut feature_maps: Vec<_> = feature_maps
                    .into_iter()
                    .filter_map(|(layer_index, output)| match layer_index {
                        NodeKey::Index(index) => Some((index, output)),
                        NodeKey::Input => None,
                    })
                    .collect();
                feature_maps.sort_by_cached_key(|(layer_index, _output)| *layer_index);
                debug_assert!(feature_maps
                    .iter()
                    .enumerate()
                    .all(|(expect_index, (layer_index, _output))| expect_index == *layer_index));
                feature_maps
                    .into_iter()
                    .map(|(_layer_index, output)| output)
                    .collect()
            };

            // combine features from every yolo layer
            let output =
                MultiDenseDetection::new(input_h as usize, input_w as usize, detections_iter)?;

            Ok((output, feature_maps))
        }
    }

    #[derive(Debug, TensorLike)]
    pub enum LayerOutput {
        Tensor(Tensor),
        Yolo(DenseDetection),
    }

    impl LayerOutput {
        pub fn tensor(self) -> Option<Tensor> {
            match self {
                Self::Tensor(tensor) => Some(tensor),
                _ => None,
            }
        }

        pub fn yolo(self) -> Option<DenseDetection> {
            match self {
                Self::Yolo(output) => Some(output),
                _ => None,
            }
        }
    }

    impl From<Tensor> for LayerOutput {
        fn from(from: Tensor) -> Self {
            Self::Tensor(from)
        }
    }

    impl From<DenseDetection> for LayerOutput {
        fn from(from: DenseDetection) -> Self {
            Self::Yolo(from)
        }
    }
}

mod layer {
    use super::*;

    macro_rules! declare_tch_layer {
        ($name:ident, $node:ty, $weights:ty) => {
            #[derive(Debug)]
            pub struct $name {
                pub node: $node,
                pub weights: $weights,
            }
        };
        ($name:ident, $node:ty) => {
            #[derive(Debug)]
            pub struct $name {
                pub node: $node,
            }
        };
    }

    #[derive(Debug, AsRefStr)]
    pub enum Layer {
        Input(InputLayer),
        Connected(ConnectedLayer),
        Convolutional(ConvolutionalLayer),
        Route(RouteLayer),
        Shortcut(ShortcutLayer),
        MaxPool(MaxPoolLayer),
        UpSample(UpSampleLayer),
        BatchNorm(BatchNormLayer),
        Yolo(YoloLayer),
        GaussianYolo(GaussianYoloLayer),
    }

    impl Layer {
        pub fn input_shape(&self) -> ShapeList {
            match self {
                Self::Input(_layer) => ShapeList::None,
                Self::Connected(layer) => ShapeList::SingleFlat(layer.node.input_shape),
                Self::Convolutional(layer) => ShapeList::SingleHwc(layer.node.input_shape),
                Self::Route(layer) => ShapeList::MultipleHwc(layer.node.input_shape.clone()),
                Self::Shortcut(layer) => ShapeList::MultipleHwc(layer.node.input_shape.clone()),
                Self::MaxPool(layer) => ShapeList::SingleHwc(layer.node.input_shape),
                Self::UpSample(layer) => ShapeList::SingleHwc(layer.node.input_shape),
                Self::BatchNorm(layer) => ShapeList::SingleHwc(layer.node.inout_shape),
                Self::Yolo(layer) => ShapeList::SingleHwc(layer.node.inout_shape),
                Self::GaussianYolo(layer) => ShapeList::SingleHwc(layer.node.inout_shape),
            }
        }

        pub fn output_shape(&self) -> Shape {
            match self {
                Self::Input(layer) => layer.node.output_shape,
                Self::Connected(layer) => Shape::Dim1(layer.node.output_shape),
                Self::Convolutional(layer) => Shape::Dim3(layer.node.output_shape),
                Self::Route(layer) => Shape::Dim3(layer.node.output_shape),
                Self::Shortcut(layer) => Shape::Dim3(layer.node.output_shape),
                Self::MaxPool(layer) => Shape::Dim3(layer.node.output_shape),
                Self::UpSample(layer) => Shape::Dim3(layer.node.output_shape),
                Self::BatchNorm(layer) => Shape::Dim3(layer.node.inout_shape),
                Self::Yolo(layer) => Shape::Dim3(layer.node.inout_shape),
                Self::GaussianYolo(layer) => Shape::Dim3(layer.node.inout_shape),
            }
        }

        pub fn from_indexes(&self) -> InputKeys {
            match self {
                Self::Input(_layer) => InputKeys::None,
                Self::Connected(layer) => InputKeys::Single(layer.node.from_indexes),
                Self::Convolutional(layer) => InputKeys::Single(layer.node.from_indexes),
                Self::Route(layer) => InputKeys::Multiple(layer.node.from_indexes.clone()),
                Self::Shortcut(layer) => InputKeys::Multiple(layer.node.from_indexes.clone()),
                Self::MaxPool(layer) => InputKeys::Single(layer.node.from_indexes),
                Self::UpSample(layer) => InputKeys::Single(layer.node.from_indexes),
                Self::BatchNorm(layer) => InputKeys::Single(layer.node.from_indexes),
                Self::Yolo(layer) => InputKeys::Single(layer.node.from_indexes),
                Self::GaussianYolo(layer) => InputKeys::Single(layer.node.from_indexes),
            }
        }

        pub fn forward_t(&mut self, xs: TensorList, train: bool) -> Result<LayerOutput> {
            let output: LayerOutput = match self {
                Layer::Input(layer) => layer.forward(xs.single().unwrap())?.into(),
                Layer::Connected(layer) => layer.forward_t(xs.single().unwrap(), train).into(),
                Layer::Convolutional(layer) => layer.forward_t(xs.single().unwrap(), train).into(),
                Layer::Route(layer) => layer.forward(xs.multiple().unwrap()).into(),
                Layer::Shortcut(layer) => layer.forward(xs.multiple().unwrap()).into(),
                Layer::MaxPool(layer) => layer.forward(xs.single().unwrap()).into(),
                Layer::UpSample(layer) => layer.forward(xs.single().unwrap()).into(),
                Layer::BatchNorm(layer) => layer.forward_t(xs.single().unwrap(), train).into(),
                Layer::Yolo(layer) => layer.forward(xs.single().unwrap())?.into(),
                Layer::GaussianYolo(layer) => layer.forward(xs.single().unwrap())?.into(),
            };
            Ok(output)
        }
    }

    #[derive(Debug)]
    pub struct InputLayer {
        pub node: InputNode,
    }

    declare_tch_layer!(ConnectedLayer, ConnectedNode, ConnectedWeights);
    declare_tch_layer!(ConvolutionalLayer, ConvolutionalNode, ConvolutionalWeights);
    declare_tch_layer!(BatchNormLayer, BatchNormNode, BatchNormWeights);
    declare_tch_layer!(ShortcutLayer, ShortcutNode, ShortcutWeights);
    declare_tch_layer!(RouteLayer, RouteNode, RouteWeights);
    declare_tch_layer!(MaxPoolLayer, MaxPoolNode, MaxPoolWeights);
    declare_tch_layer!(UpSampleLayer, UpSampleNode, UpSampleWeights);
    declare_tch_layer!(YoloLayer, YoloNode, YoloWeights);
    declare_tch_layer!(GaussianYoloLayer, GaussianYoloNode, GaussianYoloWeights);

    impl From<InputLayer> for Layer {
        fn from(from: InputLayer) -> Self {
            Self::Input(from)
        }
    }

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

    impl From<BatchNormLayer> for Layer {
        fn from(from: BatchNormLayer) -> Self {
            Self::BatchNorm(from)
        }
    }

    impl From<ShortcutLayer> for Layer {
        fn from(from: ShortcutLayer) -> Self {
            Self::Shortcut(from)
        }
    }

    impl From<RouteLayer> for Layer {
        fn from(from: RouteLayer) -> Self {
            Self::Route(from)
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

    impl From<GaussianYoloLayer> for Layer {
        fn from(from: GaussianYoloLayer) -> Self {
            Self::GaussianYolo(from)
        }
    }

    impl InputLayer {
        pub fn from_node<'p>(_path: impl Borrow<nn::Path<'p>>, from: &InputNode) -> Result<Self> {
            Ok(Self {
                node: from.to_owned(),
            })
        }

        pub fn from_darknet<'p>(
            _path: impl Borrow<nn::Path<'p>>,
            from: &darknet::InputLayer,
        ) -> Result<Self> {
            let darknet::InputLayer { node, .. } = from;
            Ok(Self {
                node: node.to_owned(),
            })
        }

        pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
            let Self {
                node: InputNode { output_shape, .. },
                ..
            } = self;

            match output_shape {
                Shape::Dim3(hwc) => {
                    let [expect_h, expect_w, expect_c] = *hwc;
                    let (_b, c, h, w) = xs.size4()?;
                    ensure!(
                        c as usize == expect_c && h as usize == expect_h && w as usize == expect_w,
                        "input shape mismatch"
                    );
                }
                Shape::Dim1(expect_size) => {
                    let (_b, c) = xs.size2()?;
                    ensure!(c as usize == *expect_size, "input shape mismatch");
                }
            }

            Ok(xs.shallow_clone())
        }
    }

    impl ConnectedLayer {
        pub fn from_node<'p>(
            path: impl Borrow<nn::Path<'p>>,
            from: &ConnectedNode,
        ) -> Result<Self> {
            let path = path.borrow();
            let ConnectedNode {
                input_shape,
                output_shape,
                config:
                    config::Connected {
                        batch_normalize, ..
                    },
                ..
            } = *from;
            let input_shape = input_shape as i64;
            let output_shape = output_shape as i64;

            let linear = nn::linear(
                path,
                input_shape,
                output_shape,
                nn::LinearConfig {
                    bias: true,
                    ..Default::default()
                },
            );

            let batch_norm = if batch_normalize {
                let batch_norm = nn::batch_norm1d(
                    path,
                    output_shape,
                    nn::BatchNormConfig {
                        momentum: 0.05,
                        eps: 0.00001,
                        ..Default::default()
                    },
                );

                Some(batch_norm)
            } else {
                None
            };

            Ok(ConnectedLayer {
                node: from.clone(),
                weights: ConnectedWeights { linear, batch_norm },
            })
        }

        pub fn from_darknet<'p>(
            path: impl Borrow<nn::Path<'p>>,
            from: &darknet::ConnectedLayer,
        ) -> Result<Self> {
            let path = path.borrow();
            let darknet::ConnectedLayer {
                node:
                    ConnectedNode {
                        input_shape,
                        output_shape,
                        ..
                    },
                weights:
                    darknet::ConnectedWeights {
                        ref weights,
                        ref biases,
                        ref scales,
                    },
            } = *from;

            let input_shape = input_shape as i64;
            let output_shape = output_shape as i64;

            let linear = {
                let mut linear = nn::linear(
                    path,
                    input_shape,
                    output_shape,
                    nn::LinearConfig {
                        bias: true,
                        ..Default::default()
                    },
                );
                linear
                    .ws
                    .replace(weights.as_slice().unwrap(), &[output_shape, input_shape]);
                linear
                    .bs
                    .replace(biases.as_slice().unwrap(), &[output_shape]);
                linear
            };

            let batch_norm = scales.as_ref().map(|scales| {
                let darknet::ScaleWeights {
                    scales,
                    rolling_mean,
                    rolling_variance,
                } = scales;

                let mut batch_norm = nn::batch_norm1d(
                    path,
                    output_shape,
                    nn::BatchNormConfig {
                        momentum: 0.05,
                        eps: 0.00001,
                        ..Default::default()
                    },
                );
                batch_norm
                    .running_mean
                    .replace(rolling_mean.as_slice().unwrap(), &[output_shape]);
                batch_norm
                    .running_var
                    .replace(rolling_variance.as_slice().unwrap(), &[output_shape]);
                batch_norm
                    .ws
                    .replace(scales.as_slice().unwrap(), &[output_shape]);

                batch_norm
            });

            Ok(ConnectedLayer {
                node: from.node.clone(),
                weights: ConnectedWeights { linear, batch_norm },
            })
        }

        pub fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
            let Self {
                node:
                    ConnectedNode {
                        config: config::Connected { activation, .. },
                        ..
                    },
                weights:
                    ConnectedWeights {
                        ref linear,
                        ref batch_norm,
                    },
                ..
            } = *self;

            let xs = xs.apply(linear);
            let xs = match batch_norm {
                Some(batch_norm) => xs.apply_t(batch_norm, train),
                None => xs,
            };

            xs.activation(activation.into())
        }
    }

    impl ConvolutionalLayer {
        pub fn from_node<'p>(
            path: impl Borrow<nn::Path<'p>>,
            from: &ConvolutionalNode,
            layer_index: usize,
            collected: &IndexMap<NodeKey, Layer>,
        ) -> Result<Self> {
            let path = path.borrow();
            let ConvolutionalNode {
                config:
                    config::Convolutional {
                        size,
                        stride_y,
                        stride_x,
                        padding,
                        groups,
                        share_index,
                        batch_normalize,
                        ..
                    },

                input_shape,
                output_shape,
                ..
            } = *from;

            {
                let [_in_w, _in_h, in_c] = input_shape;
                ensure!(
                    in_c % groups == 0,
                    "the input channels is not multiple of groups"
                );
            }

            ensure!(stride_y == stride_x, "stride_y must be equal to stride_x");
            let stride = stride_y as i64;

            let weights = match share_index {
                Some(share_index) => {
                    let share_index = share_index
                        .to_absolute(layer_index)
                        .ok_or_else(|| format_err!("invalid layer index"))?;
                    match &collected[share_index] {
                        Layer::Convolutional(target_layer) => {
                            let ConvolutionalLayer {
                                weights: ConvolutionalWeights { shared },
                                ..
                            } = target_layer;
                            ConvolutionalWeights {
                                shared: shared.clone(),
                            }
                        }
                        _ => bail!("share_index must point to convolution layer"),
                    }
                }
                None => {
                    let [_h, _w, in_c] = input_shape;
                    let [_h, _w, out_c] = output_shape;
                    let in_c = in_c as i64;
                    let out_c = out_c as i64;

                    let conv = nn::conv2d(
                        path,
                        in_c,
                        out_c,
                        size as i64,
                        nn::ConvConfig {
                            stride,
                            padding: padding as i64,
                            groups: groups as i64,
                            bias: true,
                            ..Default::default()
                        },
                    );
                    debug_assert!(matches!(conv.bs, Some(_)));

                    let batch_norm = if batch_normalize {
                        Some(nn::batch_norm2d(
                            path,
                            out_c,
                            nn::BatchNormConfig {
                                momentum: 0.1,
                                eps: 0.00001,
                                ..Default::default()
                            },
                        ))
                    } else {
                        None
                    };

                    ConvolutionalWeights {
                        shared: Arc::new(Mutex::new(ConvolutionalWeightsShared {
                            conv,
                            batch_norm,
                        })),
                    }
                }
            };

            Ok(ConvolutionalLayer {
                node: from.clone(),
                weights,
            })
        }

        pub fn from_darknet<'p>(
            path: impl Borrow<nn::Path<'p>>,
            from: &darknet::ConvolutionalLayer,
            collected: &IndexMap<NodeKey, Layer>,
        ) -> Result<Self> {
            let path = path.borrow();
            let darknet::ConvolutionalLayer {
                node:
                    ConvolutionalNode {
                        ref config,
                        input_shape,
                        output_shape,
                        ..
                    },
                ref weights,
                ..
            } = *from;

            let config::Convolutional {
                size,
                stride_y,
                stride_x,
                padding,
                groups,
                ..
            } = *config;

            let stride = if stride_y == stride_x {
                stride_y as i64
            } else {
                bail!("stride_y must be equal to stride_x")
            };

            let weights = match *weights {
                darknet::ConvolutionalWeights::Ref { share_index } => {
                    match &collected[share_index] {
                        Layer::Convolutional(target_layer) => {
                            let ConvolutionalLayer {
                                weights: ConvolutionalWeights { shared },
                                ..
                            } = target_layer;
                            ConvolutionalWeights {
                                shared: shared.clone(),
                            }
                        }
                        _ => bail!("share_index must point to convolution layer"),
                    }
                }
                darknet::ConvolutionalWeights::Owned {
                    ref biases,
                    ref scales,
                    ref weights,
                } => {
                    let [_h, _w, in_c] = input_shape;
                    let [_h, _w, out_c] = output_shape;
                    let in_c = in_c as i64;
                    let out_c = out_c as i64;
                    let permuted_weights = weights
                        .view()
                        .permuted_axes([1, 0, 3, 2])
                        .as_standard_layout()
                        .into_owned();
                    let kernel_shape = {
                        let [c1, c2, s1, s2] = if let [c1, c2, s1, s2] = *permuted_weights.shape() {
                            [c1, c2, s1, s2]
                        } else {
                            unreachable!()
                        };
                        [c1 as i64, c2 as i64, s1 as i64, s2 as i64]
                    };

                    let mut conv = nn::conv2d(
                        path,
                        in_c,
                        out_c,
                        size as i64,
                        nn::ConvConfig {
                            stride,
                            padding: padding as i64,
                            groups: groups as i64,
                            bias: true,
                            ..Default::default()
                        },
                    );

                    debug_assert!(matches!(conv.bs, Some(_)));
                    conv.ws
                        .replace(permuted_weights.as_slice().unwrap(), &kernel_shape);

                    if let Some(bs) = &mut conv.bs {
                        bs.replace(biases.as_slice().unwrap(), &[out_c]);
                    }
                    // conv.bs
                    // .as_mut()
                    // .map(|bs| bs.replace(biases.as_slice().unwrap(), &[out_c]));

                    let batch_norm = scales.as_ref().map(|scales| {
                        let darknet::ScaleWeights {
                            scales,
                            rolling_mean,
                            rolling_variance,
                        } = scales;

                        let mut batch_norm = nn::batch_norm2d(
                            path,
                            out_c,
                            nn::BatchNormConfig {
                                momentum: 0.1,
                                eps: 0.00001,
                                ..Default::default()
                            },
                        );
                        batch_norm
                            .running_mean
                            .replace(rolling_mean.as_slice().unwrap(), &[out_c]);
                        batch_norm
                            .running_var
                            .replace(rolling_variance.as_slice().unwrap(), &[out_c]);
                        batch_norm.ws.replace(scales.as_slice().unwrap(), &[out_c]);

                        batch_norm
                    });

                    ConvolutionalWeights {
                        shared: Arc::new(Mutex::new(ConvolutionalWeightsShared {
                            conv,
                            batch_norm,
                        })),
                    }
                }
            };

            Ok(ConvolutionalLayer {
                node: from.node.clone(),
                weights,
            })
        }

        pub fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
            let Self {
                node:
                    ConvolutionalNode {
                        config: config::Convolutional { activation, .. },
                        ..
                    },
                weights: ConvolutionalWeights { ref shared, .. },
                ..
            } = *self;

            let ConvolutionalWeightsShared { conv, batch_norm } = &*shared.lock().unwrap();

            let xs = xs.apply(conv);
            let xs = match batch_norm {
                Some(batch_norm) => xs.apply_t(batch_norm, train),
                None => xs,
            };

            xs.activation(activation.into())
        }
    }

    impl BatchNormLayer {
        pub fn from_node<'p>(
            path: impl Borrow<nn::Path<'p>>,
            from: &BatchNormNode,
        ) -> Result<Self> {
            let path = path.borrow();
            let BatchNormNode {
                inout_shape: [_h, _w, in_c],
                ..
            } = *from;

            let in_c = in_c as i64;

            let batch_norm = nn::batch_norm2d(
                path,
                in_c,
                nn::BatchNormConfig {
                    momentum: 0.1,
                    eps: 0.00001,
                    ..Default::default()
                },
            );

            Ok(BatchNormLayer {
                node: from.clone(),
                weights: BatchNormWeights { batch_norm },
            })
        }

        pub fn from_darknet<'p>(
            path: impl Borrow<nn::Path<'p>>,
            from: &darknet::BatchNormLayer,
        ) -> Result<Self> {
            let path = path.borrow();
            let darknet::BatchNormLayer {
                node:
                    BatchNormNode {
                        inout_shape: [_h, _w, in_c],
                        ..
                    },
                weights:
                    darknet::BatchNormWeights {
                        ref biases,
                        ref scales,
                        ref rolling_mean,
                        ref rolling_variance,
                        ..
                    },
                ..
            } = *from;

            let in_c = in_c as i64;

            let mut batch_norm = nn::batch_norm2d(
                path,
                in_c,
                nn::BatchNormConfig {
                    momentum: 0.1,
                    eps: 0.00001,
                    ..Default::default()
                },
            );
            batch_norm
                .running_mean
                .replace(rolling_mean.as_slice().unwrap(), &[in_c]);
            batch_norm
                .running_var
                .replace(rolling_variance.as_slice().unwrap(), &[in_c]);
            batch_norm.ws.replace(scales.as_slice().unwrap(), &[in_c]);
            batch_norm.bs.replace(biases.as_slice().unwrap(), &[in_c]);

            Ok(BatchNormLayer {
                node: from.node.clone(),
                weights: BatchNormWeights { batch_norm },
            })
        }

        pub fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
            let BatchNormLayer {
                weights: BatchNormWeights { batch_norm },
                ..
            } = self;
            xs.apply_t(batch_norm, train)
        }
    }

    impl ShortcutLayer {
        pub fn from_node<'p>(path: impl Borrow<nn::Path<'p>>, node: &ShortcutNode) -> Result<Self> {
            let path = path.borrow();
            let ShortcutNode {
                config: config::Shortcut { weights_type, .. },
                ref from_indexes,
                ref input_shape,
                output_shape,
                ..
            } = *node;

            let [out_h, out_w, out_c] = output_shape;
            let zero_paddings: Vec<_> = input_shape
                .iter()
                .cloned()
                .enumerate()
                .map(|(index, [_in_h, _in_w, in_c])| {
                    if in_c < out_c {
                        let zeros = path.zeros_no_train(
                            &format!("zero_padding_{}", index),
                            &[(out_c - in_c) as i64, out_h as i64, out_w as i64],
                        );
                        Some(zeros)
                    } else {
                        None
                    }
                })
                .collect();

            let num_features = from_indexes.len() as i64;
            let weights_kind = match weights_type {
                WeightsType::None => ShortcutWeightsKind::None,
                WeightsType::PerFeature => {
                    let weights = path.zeros("weights", &[num_features]);
                    ShortcutWeightsKind::PerFeature(weights)
                }
                WeightsType::PerChannel => {
                    let weights = path.zeros("weights", &[num_features, out_c as i64]);
                    ShortcutWeightsKind::PerChannel(weights)
                }
            };

            Ok(ShortcutLayer {
                node: node.clone(),
                weights: ShortcutWeights {
                    zero_paddings,
                    weights_kind,
                },
            })
        }

        pub fn from_darknet<'p>(
            path: impl Borrow<nn::Path<'p>>,
            from: &darknet::ShortcutLayer,
        ) -> Result<Self> {
            let path = path.borrow();
            let darknet::ShortcutLayer {
                node:
                    ShortcutNode {
                        ref from_indexes,
                        ref input_shape,
                        output_shape,
                        ..
                    },
                ref weights,
                ..
            } = *from;

            let [out_h, out_w, out_c] = output_shape;
            let zero_paddings: Vec<_> = input_shape
                .iter()
                .cloned()
                .enumerate()
                .map(|(index, [_in_h, _in_w, in_c])| {
                    if in_c < out_c {
                        let zeros = path.zeros_no_train(
                            &format!("zero_padding_{}", index),
                            &[(out_c - in_c) as i64, out_h as i64, out_w as i64],
                        );
                        Some(zeros)
                    } else {
                        None
                    }
                })
                .collect();

            let num_features = from_indexes.len() as i64;
            let weights_kind = match weights {
                darknet::ShortcutWeights::None => ShortcutWeightsKind::None,
                darknet::ShortcutWeights::PerFeature(from_weights) => {
                    let weights_shape = [num_features];
                    let mut to_weights = path.zeros("weights", &weights_shape);
                    to_weights.replace(from_weights.as_slice().unwrap(), &weights_shape);
                    ShortcutWeightsKind::PerFeature(to_weights)
                }
                darknet::ShortcutWeights::PerChannel(from_weights) => {
                    let weights_shape = [num_features, out_c as i64];
                    let mut to_weights = path.zeros("weights", &weights_shape);
                    to_weights.replace(from_weights.as_slice().unwrap(), &weights_shape);
                    ShortcutWeightsKind::PerChannel(to_weights)
                }
            };

            Ok(ShortcutLayer {
                node: from.node.clone(),
                weights: ShortcutWeights {
                    zero_paddings,
                    weights_kind,
                },
            })
        }

        pub fn forward<T>(&self, tensors: &[T]) -> Tensor
        where
            T: Borrow<Tensor>,
        {
            let Self {
                node:
                    ShortcutNode {
                        config:
                            config::Shortcut {
                                weights_normalization,
                                ..
                            },
                        ref from_indexes,
                        output_shape: [_h, _w, out_c],
                        ..
                    },
                weights:
                    ShortcutWeights {
                        ref zero_paddings,
                        ref weights_kind,
                    },
                ..
            } = *self;

            let out_c = out_c as i64;

            // pad or truncate channels
            let tensors: Vec<_> = zero_paddings
                .iter()
                .zip_eq(tensors.iter())
                .map(|(zero_padding, tensor)| {
                    // assume [batch, channel, height, width] shape
                    let tensor = tensor.borrow();
                    match zero_padding {
                        Some(zeros) => Tensor::cat(&[tensor, &zeros], 1),
                        None => tensor.narrow(1, 0, out_c),
                    }
                })
                .collect();

            // stack input tensors
            // becomes shape [batch, from_index, channel, height, width]
            let tensor = Tensor::stack(&tensors, 1);

            // scale by weights
            // becomes shape [batch, channel, height, width]
            let num_features = from_indexes.len() as i64;

            match weights_kind {
                ShortcutWeightsKind::None => tensor.sum1(&[1], false, tensor.kind()),
                ShortcutWeightsKind::PerFeature(weights) => {
                    let weights = match weights_normalization {
                        WeightsNormalization::None => weights.shallow_clone(),
                        WeightsNormalization::Relu => {
                            let relu = weights.relu();
                            &relu / (relu.sum(relu.kind()) + 0.0001)
                        }
                        WeightsNormalization::Softmax => weights.softmax(0, weights.kind()),
                    };

                    let weights = weights.view([1, num_features, 1, 1]).expand_as(&tensor);
                    (&tensor * weights).sum1(&[1], false, tensor.kind())
                }
                ShortcutWeightsKind::PerChannel(weights) => {
                    let weights = match weights_normalization {
                        WeightsNormalization::None => weights.shallow_clone(),
                        WeightsNormalization::Relu => {
                            // assume weights tensor has shape [num_features, num_channels]
                            let relu = weights.relu();
                            let sum = relu.sum1(&[0], true, relu.kind()).expand_as(&relu) + 0.0001;
                            relu / sum
                        }
                        WeightsNormalization::Softmax => weights.softmax(0, weights.kind()),
                    };

                    let weights = weights.view([1, num_features, out_c, 1]).expand_as(&tensor);

                    (&tensor * weights).sum1(&[1], false, tensor.kind())
                }
            }
        }
    }

    impl RouteLayer {
        pub fn from_node<'p>(_path: impl Borrow<nn::Path<'p>>, from: &RouteNode) -> Result<Self> {
            let RouteNode {
                config: config::Route { group, .. },
                ref input_shape,
                ..
            } = *from;

            let num_groups = group.num_groups();
            let group_id = group.group_id();

            let group_ranges: Vec<_> = input_shape
                .iter()
                .cloned()
                .map(|[_h, _w, c]| {
                    debug_assert_eq!(c % num_groups, 0);
                    let group_size = c / num_groups;
                    let channel_begin = group_size * group_id;
                    let channel_end = channel_begin + group_size;
                    (channel_begin as i64, channel_end as i64)
                })
                .collect();

            Ok(RouteLayer {
                node: from.clone(),
                weights: RouteWeights { group_ranges },
            })
        }

        pub fn from_darknet<'p>(
            _path: impl Borrow<nn::Path<'p>>,
            from: &darknet::RouteLayer,
        ) -> Result<Self> {
            let darknet::RouteLayer {
                node:
                    RouteNode {
                        config: config::Route { group, .. },
                        ref input_shape,
                        ..
                    },
                ..
            } = *from;

            let num_groups = group.num_groups();
            let group_id = group.group_id();

            let group_ranges: Vec<_> = input_shape
                .iter()
                .cloned()
                .map(|[_h, _w, c]| {
                    debug_assert_eq!(c % num_groups, 0);
                    let group_size = c / num_groups;
                    let channel_begin = group_size * group_id;
                    let channel_end = channel_begin + group_size;
                    (channel_begin as i64, channel_end as i64)
                })
                .collect();

            Ok(RouteLayer {
                node: from.node.clone(),
                weights: RouteWeights { group_ranges },
            })
        }

        pub fn forward<T>(&self, tensors: &[T]) -> Tensor
        where
            T: Borrow<Tensor>,
        {
            let Self {
                weights: RouteWeights { group_ranges },
                ..
            } = self;

            let sliced: Vec<_> = tensors
                .iter()
                .zip_eq(group_ranges.iter().cloned())
                .map(|(xs, (channel_begin, channel_end))| {
                    // assume [batch, channel, height, width] shape
                    let length = channel_end - channel_begin;
                    xs.borrow().narrow(1, channel_begin, length)
                })
                .collect();

            Tensor::cat(&sliced, 1)
        }
    }

    impl MaxPoolLayer {
        pub fn from_node<'p>(_path: impl Borrow<nn::Path<'p>>, from: &MaxPoolNode) -> Result<Self> {
            {
                let MaxPoolNode {
                    config:
                        config::MaxPool {
                            stride_x,
                            stride_y,
                            size,
                            padding,
                            maxpool_depth,
                            ..
                        },
                    ..
                } = *from;

                let stride_y = stride_y as i64;
                let stride_x = stride_x as i64;
                let size = size as i64;
                let padding = padding as i64;

                ensure!(padding % 2 == 0, "padding must be even");
                ensure!(!maxpool_depth, "maxpool_depth is not implemented");

                Ok(MaxPoolLayer {
                    node: from.clone(),
                    weights: MaxPoolWeights {
                        size,
                        stride_y,
                        stride_x,
                        // torch padding (one-side padding) = darknet padding / 2 (two-side padding)
                        padding: padding / 2,
                    },
                })
            }
        }

        pub fn from_darknet<'p>(
            _path: impl Borrow<nn::Path<'p>>,
            from: &darknet::MaxPoolLayer,
        ) -> Result<Self> {
            let darknet::MaxPoolLayer {
                node:
                    MaxPoolNode {
                        config:
                            config::MaxPool {
                                stride_x,
                                stride_y,
                                size,
                                padding,
                                maxpool_depth,
                                ..
                            },
                        ..
                    },
                ..
            } = *from;

            let stride_y = stride_y as i64;
            let stride_x = stride_x as i64;
            let size = size as i64;
            let padding = padding as i64;

            ensure!(padding % 2 == 0, "padding must be even");
            ensure!(!maxpool_depth, "maxpool_depth is not implemented");

            Ok(MaxPoolLayer {
                node: from.node.clone(),
                weights: MaxPoolWeights {
                    size,
                    stride_y,
                    stride_x,
                    // torch padding (one-side padding) = darknet padding / 2 (two-side padding)
                    padding: padding / 2,
                },
            })
        }

        pub fn forward(&self, xs: &Tensor) -> Tensor {
            let Self {
                weights:
                    MaxPoolWeights {
                        size,
                        stride_x,
                        stride_y,
                        padding,
                    },
                ..
            } = *self;
            xs.max_pool2d(
                &[size, size],
                &[stride_y, stride_x],
                &[padding, padding],
                &[1, 1], // dilation
                false,   // cell_mode
            )
        }
    }

    impl UpSampleLayer {
        pub fn from_node<'p>(
            _path: impl Borrow<nn::Path<'p>>,
            from: &UpSampleNode,
        ) -> Result<Self> {
            let UpSampleNode {
                output_shape: [out_h, out_w, _c],
                ..
            } = *from;

            let out_h = out_h as i64;
            let out_w = out_w as i64;

            Ok(UpSampleLayer {
                node: from.clone(),
                weights: UpSampleWeights { out_h, out_w },
            })
        }

        pub fn from_darknet<'p>(
            _path: impl Borrow<nn::Path<'p>>,
            from: &darknet::UpSampleLayer,
        ) -> Result<Self> {
            let darknet::UpSampleLayer {
                node:
                    UpSampleNode {
                        output_shape: [out_h, out_w, _c],
                        ..
                    },
                ..
            } = *from;

            let out_h = out_h as i64;
            let out_w = out_w as i64;

            Ok(UpSampleLayer {
                node: from.node.clone(),
                weights: UpSampleWeights { out_h, out_w },
            })
        }

        pub fn forward(&self, xs: &Tensor) -> Tensor {
            let Self {
                weights: UpSampleWeights { out_h, out_w },
                ..
            } = *self;
            xs.upsample_nearest2d(&[out_h, out_w], None, None)
        }
    }

    impl YoloLayer {
        pub fn from_node<'p>(_path: impl Borrow<nn::Path<'p>>, from: &YoloNode) -> Result<Self> {
            let YoloNode {
                config: config::Yolo { classes, .. },
                ..
            } = *from;
            let weights = YoloWeights {
                num_classes: classes as i64,
                cache: None,
            };

            Ok(Self {
                node: from.clone(),
                weights,
            })
        }

        pub fn from_darknet<'p>(
            _path: impl Borrow<nn::Path<'p>>,
            from: &darknet::YoloLayer,
        ) -> Result<Self> {
            let darknet::YoloLayer {
                node:
                    YoloNode {
                        config: config::Yolo { classes, .. },
                        ..
                    },
                ..
            } = *from;
            let weights = YoloWeights {
                num_classes: classes as i64,
                cache: None,
            };

            Ok(Self {
                node: from.node.clone(),
                weights,
            })
        }

        pub fn forward(&mut self, input: &Tensor) -> Result<DenseDetection> {
            let YoloCache {
                y_grids, x_grids, ..
            } = self.cache(input);
            let Self {
                node:
                    YoloNode {
                        config: config::Yolo { ref anchors, .. },
                        ..
                    },
                weights: YoloWeights { num_classes, .. },
                ..
            } = *self;

            let num_anchors = anchors.len() as i64;

            // reshape to [bsize, n_anchors, n_classes + bbox (4) + objectness (1), height, width]
            let (bsize, channels, height, width) = input.size4()?;
            debug_assert!(channels % num_anchors == 0);
            let xs = input.view([bsize, num_anchors, -1, height, width]);

            // unpack detection parameters
            let raw_x = xs.i((.., .., 0..1, .., ..));
            let raw_y = xs.i((.., .., 1..2, .., ..));
            let raw_w = xs.i((.., .., 2..3, .., ..));
            let raw_h = xs.i((.., .., 3..4, .., ..));
            let objectness = xs.i((.., .., 4..5, .., ..));
            let class = xs.i((.., .., 5..(num_classes + 5), .., ..));

            // calculate bbox
            let bbox_cy = (&raw_y + y_grids.expand_as(&raw_y)) / height as f64;
            let bbox_cx = (&raw_x + x_grids.expand_as(&raw_x)) / width as f64;
            let bbox_h = (raw_h.exp() + 0.5) / height as f64;
            let bbox_w = (raw_w.exp() + 0.5) / width as f64;

            // convert to [bsize, entries, anchors, height, width] shape
            let bbox_cy = bbox_cy.permute(&[0, 2, 1, 3, 4]).contiguous();
            let bbox_cx = bbox_cx.permute(&[0, 2, 1, 3, 4]).contiguous();
            let bbox_h = bbox_h.permute(&[0, 2, 1, 3, 4]).contiguous();
            let bbox_w = bbox_w.permute(&[0, 2, 1, 3, 4]).contiguous();
            let objectness = objectness.permute(&[0, 2, 1, 3, 4]).contiguous();
            let class = class.permute(&[0, 2, 1, 3, 4]).contiguous();

            // anchors
            let anchors: Vec<_> = anchors
                .iter()
                .cloned()
                .map(|(anchor_h, anchor_w)| {
                    GridSize::new(anchor_h as f64, anchor_w as f64).unwrap()
                })
                .collect();

            DenseDetectionInit {
                anchors,
                num_classes: num_classes as usize,
                bbox_cy,
                bbox_cx,
                bbox_h,
                bbox_w,
                objectness,
                classification: class,
            }
            .try_into()
        }

        pub fn cache(&mut self, xs: &Tensor) -> YoloCache {
            let (_bsize, _channels, height, width) = xs.size4().unwrap();
            let device = xs.device();
            let kind = xs.kind();

            let shoud_update = self
                .weights
                .cache
                .as_ref()
                .map(
                    |&YoloCache {
                         expect_height,
                         expect_width,
                         ..
                     }| !(expect_height == height && expect_width == width),
                )
                .unwrap_or(true);

            if shoud_update {
                let (y_grids, x_grids) = tch::no_grad(|| {
                    let grids = Tensor::meshgrid(&[
                        Tensor::arange(height, (kind, device)),
                        Tensor::arange(width, (kind, device)),
                    ]);

                    // stack and reshape to (batch x anchors x entry x height x width)
                    // Tensor::stack(&[&grids[0], &grids[1]], 0).view([1, 1, 2, height, width])
                    let y_grids = grids[0].view([1, 1, 1, height, width]);
                    let x_grids = grids[1].view([1, 1, 1, height, width]);

                    (y_grids, x_grids)
                });

                self.weights.cache = Some(YoloCache {
                    expect_height: height,
                    expect_width: width,
                    y_grids,
                    x_grids,
                });
            }

            self.weights.cache.as_ref().unwrap().shallow_clone()
        }
    }

    impl GaussianYoloLayer {
        pub fn from_node<'p>(
            _path: impl Borrow<nn::Path<'p>>,
            from: &GaussianYoloNode,
        ) -> Result<Self> {
            let GaussianYoloNode {
                config: config::GaussianYolo { classes, .. },
                ..
            } = *from;
            let weights = GaussianYoloWeights {
                num_classes: classes as i64,
                cache: None,
            };

            Ok(Self {
                node: from.clone(),
                weights,
            })
        }

        pub fn from_darknet<'p>(
            _path: impl Borrow<nn::Path<'p>>,
            from: &darknet::GaussianYoloLayer,
        ) -> Result<Self> {
            let darknet::GaussianYoloLayer {
                node:
                    GaussianYoloNode {
                        config: config::GaussianYolo { classes, .. },
                        ..
                    },
                ..
            } = *from;
            let weights = GaussianYoloWeights {
                num_classes: classes as i64,
                cache: None,
            };

            Ok(Self {
                node: from.node.clone(),
                weights,
            })
        }

        pub fn forward(&mut self, _input: &Tensor) -> Result<DenseDetection> {
            unimplemented!();
        }
    }
}

mod weights {
    use super::*;

    #[derive(Debug)]
    pub struct ConnectedWeights {
        pub linear: nn::Linear,
        pub batch_norm: Option<nn::BatchNorm>,
    }

    #[derive(Debug)]
    pub struct ConvolutionalWeights {
        pub shared: Arc<Mutex<ConvolutionalWeightsShared>>,
    }

    #[derive(Debug)]
    pub struct ConvolutionalWeightsShared {
        pub conv: nn::Conv2D,
        pub batch_norm: Option<nn::BatchNorm>,
    }

    #[derive(Debug)]
    pub struct MaxPoolWeights {
        pub size: i64,
        pub stride_x: i64,
        pub stride_y: i64,
        pub padding: i64,
    }

    #[derive(Debug)]
    pub struct UpSampleWeights {
        pub out_h: i64,
        pub out_w: i64,
    }

    #[derive(Debug)]
    pub struct BatchNormWeights {
        pub batch_norm: nn::BatchNorm,
    }

    #[derive(Debug)]
    pub struct ShortcutWeights {
        pub zero_paddings: Vec<Option<Tensor>>,
        pub weights_kind: ShortcutWeightsKind,
    }

    #[derive(Debug)]
    pub enum ShortcutWeightsKind {
        None,
        PerFeature(Tensor),
        PerChannel(Tensor),
    }

    #[derive(Debug)]
    pub struct RouteWeights {
        pub group_ranges: Vec<(i64, i64)>,
    }

    #[derive(Debug)]
    pub struct YoloWeights {
        pub num_classes: i64,
        pub cache: Option<YoloCache>,
    }

    #[derive(Debug)]
    pub struct GaussianYoloWeights {
        pub num_classes: i64,
        pub cache: Option<YoloCache>,
    }

    #[derive(Debug, TensorLike)]
    pub struct YoloCache {
        #[tensor_like(copy)]
        pub expect_height: i64,
        #[tensor_like(copy)]
        pub expect_width: i64,
        pub y_grids: Tensor,
        pub x_grids: Tensor,
    }
}
