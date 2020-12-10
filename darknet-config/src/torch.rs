use crate::{
    common::*,
    config::{
        Activation, ConnectedConfig, ConvolutionalConfig, DarknetConfig, MaxPoolConfig, NetConfig,
        RouteConfig, Shape, ShortcutConfig, UpSampleConfig, WeightsNormalization, WeightsType,
        YoloConfig,
    },
    darknet::{self, DarknetModel},
    graph::{
        BatchNormNode, ConnectedNode, ConvolutionalNode, GaussianYoloNode, Graph, LayerPosition,
        LayerPositionSet, MaxPoolNode, Node, RouteNode, ShapeList, ShortcutNode, UpSampleNode,
        YoloNode,
    },
};
use tch::{nn, Kind, Tensor};

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

trait TensorActivationEx {
    fn activation(&self, activation: Activation) -> Tensor;
}

impl TensorActivationEx for Tensor {
    fn activation(&self, activation: Activation) -> Tensor {
        match activation {
            Activation::Relu => self.relu(),
            Activation::Swish => self.swish(),
            Activation::Mish => self.mish(),
            Activation::HardMish => self.hardswish(),
            // Activation::NormalizeChannels => todo!(),
            // Activation::NormalizeChannelsSoftmax => todo!(),
            // Activation::NormalizeChannelsSoftmaxMaxval => todo!(),
            _ => unimplemented!(),
        }
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
        pub layers: IndexMap<usize, Layer>,
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
            config: &DarknetConfig,
        ) -> Result<Self> {
            let model = DarknetModel::from_config(config)?;
            Self::from_darknet_model(path, &model)
        }

        pub fn from_darknet_model<'p>(
            path: impl Borrow<nn::Path<'p>>,
            model: &DarknetModel,
        ) -> Result<Self> {
            let path = path.borrow();
            let DarknetModel {
                graph:
                    Graph {
                        net:
                            NetConfig {
                                classes: num_classes,
                                ..
                            },
                        ..
                    },
                ..
            } = *model;

            let layers: IndexMap<_, _> = model.layers.iter().try_fold(
                IndexMap::new(),
                |mut collected, (&layer_index, layer)| -> Result<_> {
                    let layer: Layer = match layer {
                        darknet::Layer::Connected(conf) => ConnectedLayer::new(path, conf)?.into(),
                        darknet::Layer::Convolutional(conf) => {
                            ConvolutionalLayer::new(path, conf, &collected)?.into()
                        }
                        darknet::Layer::BatchNorm(conf) => BatchNormLayer::new(path, conf)?.into(),
                        darknet::Layer::MaxPool(conf) => MaxPoolLayer::new(path, conf)?.into(),
                        darknet::Layer::UpSample(conf) => UpSampleLayer::new(path, conf)?.into(),
                        darknet::Layer::Shortcut(conf) => ShortcutLayer::new(path, conf)?.into(),
                        darknet::Layer::Route(conf) => RouteLayer::new(path, conf)?.into(),
                        darknet::Layer::Yolo(conf) => {
                            YoloLayer::new(path, conf, num_classes)?.into()
                        }
                        darknet::Layer::GaussianYolo(conf) => {
                            GaussianYoloLayer::new(path, conf, num_classes)?.into()
                        }
                    };

                    collected.insert(layer_index, layer);
                    Ok(collected)
                },
            )?;

            Ok(Self {
                graph: model.graph.clone(),
                layers,
            })
        }

        pub fn forward_t(&mut self, input: &Tensor, train: bool) -> Result<MultiDenseDetection> {
            let (_batch_size, _channels, input_h, input_w) = input.size4()?;

            let detections_iter = {
                let mut intermediate_tensors = hashmap! {
                    LayerPosition::Input => input.shallow_clone()
                };
                let mut final_tensors = hashmap! {};

                self.layers
                    .iter_mut()
                    .try_for_each(|(&layer_index, layer)| -> Result<_> {
                        // get input tensors
                        let input = match layer.from_indexes() {
                            LayerPositionSet::Single(index) => {
                                let tensor = intermediate_tensors[&index].shallow_clone();
                                TensorList::Single(tensor)
                            }
                            LayerPositionSet::Multiple(indexes) => {
                                let tensors: Vec<_> = indexes
                                    .iter()
                                    .map(|index| intermediate_tensors[&index].shallow_clone())
                                    .collect();
                                TensorList::Multiple(tensors)
                            }
                            LayerPositionSet::Empty => unreachable!(),
                        };

                        // save output
                        match layer.forward_t(input, train)? {
                            LayerOutputKind::Tensor(output) => {
                                let prev = intermediate_tensors
                                    .insert(LayerPosition::Absolute(layer_index), output);
                                debug_assert!(matches!(prev, None));
                            }
                            LayerOutputKind::Yolo(output) => {
                                let prev = final_tensors.insert(layer_index, output);
                                debug_assert!(matches!(prev, None));
                            }
                        }

                        Ok(())
                    })?;

                // pack output tensors
                let exported_iter = {
                    let mut exported: Vec<_> = final_tensors.into_iter().collect();
                    exported.sort_by_cached_key(|(layer_index, _output)| *layer_index);
                    exported.into_iter().map(|(_layer_index, output)| output)
                };

                exported_iter
            };

            // combine features from every yolo layer

            let output =
                MultiDenseDetection::new(input_h as usize, input_w as usize, detections_iter)?;

            Ok(output)
        }
    }

    #[derive(Debug)]
    pub enum LayerOutputKind {
        Tensor(Tensor),
        Yolo(DenseDetection),
    }

    impl LayerOutputKind {
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

    impl From<Tensor> for LayerOutputKind {
        fn from(from: Tensor) -> Self {
            Self::Tensor(from)
        }
    }

    impl From<DenseDetection> for LayerOutputKind {
        fn from(from: DenseDetection) -> Self {
            Self::Yolo(from)
        }
    }
}

mod layer {
    use super::*;

    macro_rules! declare_tch_layer {
        ($name:ident, $base:ty, $weights:ty) => {
            #[derive(Debug)]
            pub struct $name {
                pub base: $base,
                pub weights: $weights,
            }
        };
        ($name:ident, $base:ty) => {
            #[derive(Debug)]
            pub struct $name {
                pub base: $base,
            }
        };
    }

    #[derive(Debug)]
    pub enum Layer {
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
                Self::Connected(layer) => ShapeList::SingleFlat(layer.base.input_shape),
                Self::Convolutional(layer) => ShapeList::SingleHwc(layer.base.input_shape),
                Self::Route(layer) => ShapeList::MultipleHwc(layer.base.input_shape.clone()),
                Self::Shortcut(layer) => ShapeList::MultipleHwc(layer.base.input_shape.clone()),
                Self::MaxPool(layer) => ShapeList::SingleHwc(layer.base.input_shape),
                Self::UpSample(layer) => ShapeList::SingleHwc(layer.base.input_shape),
                Self::BatchNorm(layer) => ShapeList::SingleHwc(layer.base.inout_shape),
                Self::Yolo(layer) => ShapeList::SingleHwc(layer.base.inout_shape),
                Self::GaussianYolo(layer) => ShapeList::SingleHwc(layer.base.inout_shape),
            }
        }

        pub fn output_shape(&self) -> Shape {
            match self {
                Self::Connected(layer) => Shape::Flat(layer.base.output_shape),
                Self::Convolutional(layer) => Shape::Hwc(layer.base.output_shape),
                Self::Route(layer) => Shape::Hwc(layer.base.output_shape),
                Self::Shortcut(layer) => Shape::Hwc(layer.base.output_shape),
                Self::MaxPool(layer) => Shape::Hwc(layer.base.output_shape),
                Self::UpSample(layer) => Shape::Hwc(layer.base.output_shape),
                Self::BatchNorm(layer) => Shape::Hwc(layer.base.inout_shape),
                Self::Yolo(layer) => Shape::Hwc(layer.base.inout_shape),
                Self::GaussianYolo(layer) => Shape::Hwc(layer.base.inout_shape),
            }
        }

        pub fn from_indexes(&self) -> LayerPositionSet {
            match self {
                Self::Connected(layer) => LayerPositionSet::Single(layer.base.from_indexes),
                Self::Convolutional(layer) => LayerPositionSet::Single(layer.base.from_indexes),
                Self::Route(layer) => LayerPositionSet::Multiple(layer.base.from_indexes.clone()),
                Self::Shortcut(layer) => {
                    LayerPositionSet::Multiple(layer.base.from_indexes.clone())
                }
                Self::MaxPool(layer) => LayerPositionSet::Single(layer.base.from_indexes),
                Self::UpSample(layer) => LayerPositionSet::Single(layer.base.from_indexes),
                Self::BatchNorm(layer) => LayerPositionSet::Single(layer.base.from_indexes),
                Self::Yolo(layer) => LayerPositionSet::Single(layer.base.from_indexes),
                Self::GaussianYolo(layer) => LayerPositionSet::Single(layer.base.from_indexes),
            }
        }

        pub fn forward_t(&mut self, xs: TensorList, train: bool) -> Result<LayerOutputKind> {
            let output: LayerOutputKind = match self {
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

    declare_tch_layer!(ConnectedLayer, ConnectedNode, ConnectedWeights);
    declare_tch_layer!(ConvolutionalLayer, ConvolutionalNode, ConvolutionalWeights);
    declare_tch_layer!(BatchNormLayer, BatchNormNode, BatchNormWeights);
    declare_tch_layer!(ShortcutLayer, ShortcutNode, ShortcutWeights);
    declare_tch_layer!(RouteLayer, RouteNode, RouteWeights);
    declare_tch_layer!(MaxPoolLayer, MaxPoolNode, MaxPoolWeights);
    declare_tch_layer!(UpSampleLayer, UpSampleNode, UpSampleWeights);
    declare_tch_layer!(YoloLayer, YoloNode, YoloWeights);
    declare_tch_layer!(GaussianYoloLayer, GaussianYoloNode, GaussianYoloWeights);

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

    impl ConnectedLayer {
        pub fn new<'p>(
            path: impl Borrow<nn::Path<'p>>,
            from: &darknet::ConnectedLayer,
        ) -> Result<Self> {
            let path = path.borrow();
            let darknet::ConnectedLayer {
                base:
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
                base: from.base.clone(),
                weights: ConnectedWeights { linear, batch_norm },
            })
        }

        pub fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
            let Self {
                base:
                    ConnectedNode {
                        config: ConnectedConfig { activation, .. },
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
            let xs = xs.activation(activation);
            xs
        }
    }

    impl ConvolutionalLayer {
        pub fn new<'p>(
            path: impl Borrow<nn::Path<'p>>,
            from: &darknet::ConvolutionalLayer,
            collected: &IndexMap<usize, Layer>,
        ) -> Result<Self> {
            let path = path.borrow();
            let darknet::ConvolutionalLayer {
                base:
                    ConvolutionalNode {
                        ref config,
                        input_shape,
                        output_shape,
                        ..
                    },
                ref weights,
                ..
            } = *from;

            let ConvolutionalConfig {
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
                        let [c1, c2, s1, s2] = match permuted_weights.shape() {
                            &[c1, c2, s1, s2] => [c1, c2, s1, s2],
                            _ => unreachable!(),
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
                    conv.bs
                        .as_mut()
                        .map(|bs| bs.replace(biases.as_slice().unwrap(), &[out_c]));

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
                base: from.base.clone(),
                weights,
            })
        }

        pub fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
            let Self {
                base:
                    ConvolutionalNode {
                        config: ConvolutionalConfig { activation, .. },
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
            let xs = xs.activation(activation);
            xs
        }
    }

    impl BatchNormLayer {
        pub fn new<'p>(
            path: impl Borrow<nn::Path<'p>>,
            from: &darknet::BatchNormLayer,
        ) -> Result<Self> {
            let path = path.borrow();
            let darknet::BatchNormLayer {
                base:
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
                base: from.base.clone(),
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
        pub fn new<'p>(
            path: impl Borrow<nn::Path<'p>>,
            from: &darknet::ShortcutLayer,
        ) -> Result<Self> {
            let path = path.borrow();
            let darknet::ShortcutLayer {
                base:
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
                base: from.base.clone(),
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
                base:
                    ShortcutNode {
                        config:
                            ShortcutConfig {
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
                    let tensor = match zero_padding {
                        Some(zeros) => Tensor::cat(&[tensor, &zeros], 1),
                        None => tensor.narrow(1, 0, out_c),
                    };
                    tensor
                })
                .collect();

            // stack input tensors
            // becomes shape [batch, from_index, channel, height, width]
            let tensor = Tensor::cat(&tensors, 1);

            // scale by weights
            // becomes shape [batch, channel, height, width]
            let num_input_layers = from_indexes.len() as i64;

            let tensor = match weights_kind {
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

                    let weights = weights.view([1, num_input_layers, 1, 1]).expand_as(&tensor);
                    (&tensor * weights).sum1(&[1], false, tensor.kind())
                }
                ShortcutWeightsKind::PerChannel(weights) => {
                    let weights = match weights_normalization {
                        WeightsNormalization::None => weights.shallow_clone(),
                        WeightsNormalization::Relu => {
                            // assume weights tensor has shape [num_input_layers, num_channels]
                            let relu = weights.relu();
                            let sum = relu.sum1(&[0], true, relu.kind()).expand_as(&relu) + 0.0001;
                            relu / sum
                        }
                        WeightsNormalization::Softmax => weights.softmax(0, weights.kind()),
                    };

                    let weights = weights
                        .view([1, num_input_layers, out_c, 1])
                        .expand_as(&tensor);

                    (&tensor * weights).sum1(&[1], false, tensor.kind())
                }
            };

            tensor
        }
    }

    impl RouteLayer {
        pub fn new<'p>(
            _path: impl Borrow<nn::Path<'p>>,
            from: &darknet::RouteLayer,
        ) -> Result<Self> {
            let darknet::RouteLayer {
                base:
                    RouteNode {
                        config: RouteConfig { group, .. },
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
                base: from.base.clone(),
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
        pub fn new<'p>(
            _path: impl Borrow<nn::Path<'p>>,
            from: &darknet::MaxPoolLayer,
        ) -> Result<Self> {
            let darknet::MaxPoolLayer {
                base:
                    MaxPoolNode {
                        config:
                            MaxPoolConfig {
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

            ensure!(!maxpool_depth, "maxpool_depth is not implemented");

            Ok(MaxPoolLayer {
                base: from.base.clone(),
                weights: MaxPoolWeights {
                    size,
                    stride_y,
                    stride_x,
                    padding,
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
                &[],   // dilation
                false, // cell_mode
            )
        }
    }

    impl UpSampleLayer {
        pub fn new<'p>(
            _path: impl Borrow<nn::Path<'p>>,
            from: &darknet::UpSampleLayer,
        ) -> Result<Self> {
            let darknet::UpSampleLayer {
                base:
                    UpSampleNode {
                        output_shape: [out_h, out_w, _c],
                        ..
                    },
                ..
            } = *from;

            let out_h = out_h as i64;
            let out_w = out_w as i64;

            Ok(UpSampleLayer {
                base: from.base.clone(),
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
        pub fn new<'p>(
            _path: impl Borrow<nn::Path<'p>>,
            from: &darknet::YoloLayer,
            num_classes: u64,
        ) -> Result<Self> {
            let weights = YoloWeights {
                num_classes: num_classes as i64,
                cache: None,
            };

            Ok(Self {
                base: from.base.clone(),
                weights,
            })
        }

        pub fn forward(&mut self, input: &Tensor) -> Result<DenseDetection> {
            let YoloCache {
                y_grids, x_grids, ..
            } = self.cache(input);
            let Self {
                base:
                    YoloNode {
                        config: YoloConfig { ref anchors, .. },
                        ..
                    },
                weights: YoloWeights { num_classes, .. },
                ..
            } = *self;

            let num_anchors = anchors.len() as i64;

            // reshape to [bsize, n_anchors, n_classes + 4 + 1, height, width]
            let (bsize, channels, height, width) = input.size4()?;
            debug_assert!(channels % num_anchors == 0);
            let xs = input.view([bsize, num_anchors, -1, height, width]);

            // unpack detection parameters
            let raw_x = xs.narrow(2, 0, 1);
            let raw_y = xs.narrow(2, 1, 1);
            let raw_w = xs.narrow(2, 2, 1);
            let raw_h = xs.narrow(2, 3, 1);
            let objectness = xs.narrow(2, 4, 1);
            let class = xs.narrow(2, 5, num_classes);

            // calculate bbox
            let bbox_cy = (&raw_y + y_grids.expand_as(&raw_y)) / height as f64;
            let bbox_cx = (&raw_x + x_grids.expand_as(&raw_x)) / width as f64;
            let bbox_h = (raw_h.exp() + 0.5) / height as f64;
            let bbox_w = (raw_w.exp() + 0.5) / width as f64;

            // anchors
            let anchors: Vec<_> = anchors
                .iter()
                .cloned()
                .map(|(anchor_h, anchor_w)| GridSize::new(anchor_h as f64, anchor_w as f64))
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
        pub fn new<'p>(
            _path: impl Borrow<nn::Path<'p>>,
            from: &darknet::GaussianYoloLayer,
            num_classes: u64,
        ) -> Result<Self> {
            let weights = GaussianYoloWeights {
                num_classes: num_classes as i64,
                cache: None,
            };

            Ok(Self {
                base: from.base.clone(),
                weights,
            })
        }

        pub fn forward(&mut self, input: &Tensor) -> Result<DenseDetection> {
            todo!();
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
