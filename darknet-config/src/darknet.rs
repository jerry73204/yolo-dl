use crate::{
    common::*,
    config::{
        BatchNormConfig, Common, ConnectedConfig, ConvolutionalConfig, DarknetConfig,
        ShortcutConfig, WeightsType,
    },
    graph::{
        BatchNormNode, ConnectedNode, ConvolutionalNode, GaussianYoloNode, Graph, MaxPoolNode,
        Node, RouteNode, ShortcutNode, UpSampleNode, YoloNode,
    },
};

pub use layer::*;
pub use model::*;
pub use weights::*;

mod model {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct DarknetModel {
        pub graph: Graph,
        pub layers: IndexMap<usize, Layer>,
    }

    impl DarknetModel {
        pub fn new(graph: &Graph) -> Result<Self> {
            // aggregate all computed features
            let layers: IndexMap<_, _> = {
                graph
                    .layers
                    .iter()
                    .map(|(&layer_index, layer_node)| -> Result<_> {
                        let layer = match layer_node {
                            Node::Connected(node) => Layer::Connected(ConnectedLayer::new(node)),
                            Node::Convolutional(node) => {
                                Layer::Convolutional(ConvolutionalLayer::new(node, layer_index)?)
                            }
                            Node::Route(node) => Layer::Route(RouteLayer { node: node.clone() }),
                            Node::Shortcut(node) => Layer::Shortcut(ShortcutLayer::new(node)),
                            Node::MaxPool(node) => {
                                Layer::MaxPool(MaxPoolLayer { node: node.clone() })
                            }
                            Node::UpSample(node) => {
                                Layer::UpSample(UpSampleLayer { node: node.clone() })
                            }
                            Node::BatchNorm(node) => Layer::BatchNorm(BatchNormLayer::new(node)),
                            Node::Dropout(_node) => unimplemented!(),
                            Node::Softmax(_node) => unimplemented!(),
                            Node::Yolo(node) => Layer::Yolo(YoloLayer { node: node.clone() }),
                            Node::GaussianYolo(node) => {
                                Layer::GaussianYolo(GaussianYoloLayer { node: node.clone() })
                            }
                        };

                        Ok((layer_index, layer))
                    })
                    .try_collect()?
            };

            Ok(Self {
                graph: graph.clone(),
                layers,
            })
        }

        pub fn from_config_file<P>(config_file: P) -> Result<Self>
        where
            P: AsRef<Path>,
        {
            let config = DarknetConfig::load(config_file)?;
            let graph = Graph::from_config(&config)?;
            Self::new(&graph)
        }

        pub fn from_config(config: &DarknetConfig) -> Result<Self> {
            let graph = Graph::from_config(config)?;
            Self::new(&graph)
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
            self.graph.seen = seen;
            self.graph.cur_iteration = self.graph.net.iteration(seen);

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
}

mod layer {
    use super::*;

    macro_rules! declare_darknet_layer {
        ($name:ident, $node:ty, $weights:ty) => {
            #[derive(Debug, Clone)]
            pub struct $name {
                pub node: $node,
                pub weights: $weights,
            }
        };
        ($name:ident, $node:ty) => {
            #[derive(Debug, Clone)]
            pub struct $name {
                pub node: $node,
            }
        };
    }

    #[derive(Debug, Clone, AsRefStr)]
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
        pub fn load_weights(&mut self, reader: impl ReadBytesExt, transpose: bool) -> Result<()> {
            match self {
                Self::Connected(layer) => layer.load_weights(reader, transpose),
                Self::Convolutional(layer) => layer.load_weights(reader),
                Self::Route(_layer) => Ok(()),
                Self::Shortcut(layer) => layer.load_weights(reader),
                Self::MaxPool(_layer) => Ok(()),
                Self::UpSample(_layer) => Ok(()),
                Self::BatchNorm(layer) => layer.load_weights(reader),
                Self::Yolo(_layer) => Ok(()),
                Self::GaussianYolo(_layer) => Ok(()),
            }
        }
    }

    declare_darknet_layer!(ConnectedLayer, ConnectedNode, ConnectedWeights);
    declare_darknet_layer!(ConvolutionalLayer, ConvolutionalNode, ConvolutionalWeights);
    declare_darknet_layer!(BatchNormLayer, BatchNormNode, BatchNormWeights);
    declare_darknet_layer!(ShortcutLayer, ShortcutNode, ShortcutWeights);
    declare_darknet_layer!(RouteLayer, RouteNode);
    declare_darknet_layer!(MaxPoolLayer, MaxPoolNode);
    declare_darknet_layer!(UpSampleLayer, UpSampleNode);
    declare_darknet_layer!(YoloLayer, YoloNode);
    declare_darknet_layer!(GaussianYoloLayer, GaussianYoloNode);

    impl ConnectedLayer {
        pub fn new(node: &ConnectedNode) -> Self {
            let ConnectedNode {
                config: ConnectedConfig {
                    batch_normalize, ..
                },
                input_shape,
                output_shape,
                ..
            } = *node;
            let input_shape = input_shape as usize;
            let output_shape = output_shape as usize;

            let weights = ConnectedWeights {
                biases: Array1::from_shape_vec(input_shape, vec![0.0; input_shape]).unwrap(),
                weights: Array2::from_shape_vec(
                    [input_shape, output_shape],
                    vec![0.0; input_shape * output_shape],
                )
                .unwrap(),
                scales: if batch_normalize {
                    Some(ScaleWeights::new(output_shape as usize))
                } else {
                    None
                },
            };

            Self {
                node: node.clone(),
                weights,
            }
        }

        pub fn load_weights(
            &mut self,
            mut reader: impl ReadBytesExt,
            transpose: bool,
        ) -> Result<()> {
            let Self {
                node:
                    ConnectedNode {
                        config:
                            ConnectedConfig {
                                common:
                                    Common {
                                        dont_load,
                                        dont_load_scales,
                                        ..
                                    },
                                ..
                            },
                        input_shape,
                        output_shape,
                        ..
                    },
                weights:
                    ConnectedWeights {
                        ref mut biases,
                        ref mut weights,
                        ref mut scales,
                    },
                ..
            } = *self;

            if dont_load {
                return Ok(());
            }

            reader.read_f32_into::<LittleEndian>(biases.as_slice_mut().unwrap())?;
            reader.read_f32_into::<LittleEndian>(weights.as_slice_mut().unwrap())?;

            if transpose {
                crate::utils::transpose_matrix(
                    weights.as_slice_mut().unwrap(),
                    input_shape as usize,
                    output_shape as usize,
                )?;
            }

            if let (Some(scales), false) = (scales, dont_load_scales) {
                scales.load_weights(reader)?;
            }

            Ok(())
        }
    }

    impl ConvolutionalLayer {
        pub fn new(node: &ConvolutionalNode, layer_index: usize) -> Result<Self> {
            let ConvolutionalNode {
                config:
                    ConvolutionalConfig {
                        share_index,
                        filters,
                        batch_normalize,
                        size,
                        groups,
                        ..
                    },
                input_shape: [_, _, in_c],
                ..
            } = *node;

            ensure!(
                in_c % groups == 0,
                "the input channels is not multiple of groups"
            );

            let weights = if let Some(share_index) = share_index {
                let share_index = share_index
                    .to_absolute(layer_index)
                    .ok_or_else(|| format_err!("invalid layer index"))?;
                ConvolutionalWeights::Ref { share_index }
            } else {
                let weights_shape = {
                    let [s1, s2, s3, s4] = [in_c / groups, filters, size, size];
                    [s1 as usize, s2 as usize, s3 as usize, s4 as usize]
                };
                let weights = Array4::from_shape_vec(
                    weights_shape,
                    vec![0.0; weights_shape.iter().cloned().product()],
                )
                .unwrap();
                let biases =
                    Array1::from_shape_vec(filters as usize, vec![0.0; filters as usize]).unwrap();
                let scales = if batch_normalize {
                    Some(ScaleWeights::new(filters as usize))
                } else {
                    None
                };

                ConvolutionalWeights::Owned {
                    biases,
                    weights,
                    scales,
                }
            };

            Ok(Self {
                node: node.clone(),
                weights,
            })
        }

        pub fn load_weights(&mut self, mut reader: impl ReadBytesExt) -> Result<()> {
            let Self {
                node:
                    ConvolutionalNode {
                        config:
                            ConvolutionalConfig {
                                groups,
                                size,
                                filters,
                                flipped,
                                common:
                                    Common {
                                        dont_load,
                                        dont_load_scales,
                                        ..
                                    },
                                ..
                            },
                        input_shape: [_h, _w, in_c],
                        ..
                    },
                ref mut weights,
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
                    reader.read_f32_into::<LittleEndian>(biases.as_slice_mut().unwrap())?;

                    if let (Some(scales), false) = (scales, dont_load_scales) {
                        scales.load_weights(&mut reader)?;
                    }

                    reader.read_f32_into::<LittleEndian>(weights.as_slice_mut().unwrap())?;

                    if flipped {
                        crate::utils::transpose_matrix(
                            weights.as_slice_mut().unwrap(),
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
        pub fn new(node: &BatchNormNode) -> Self {
            let [_h, _w, channels] = node.inout_shape;
            let channels = channels as usize;

            let biases = Array1::from_shape_vec(channels, vec![0.0; channels]).unwrap();
            let scales = Array1::from_shape_vec(channels, vec![0.0; channels]).unwrap();
            let rolling_mean = Array1::from_shape_vec(channels, vec![0.0; channels]).unwrap();
            let rolling_variance = Array1::from_shape_vec(channels, vec![0.0; channels]).unwrap();

            let weights = BatchNormWeights {
                biases,
                scales,
                rolling_mean,
                rolling_variance,
            };

            Self {
                node: node.clone(),
                weights,
            }
        }

        pub fn load_weights(&mut self, mut reader: impl ReadBytesExt) -> Result<()> {
            let Self {
                node:
                    BatchNormNode {
                        config:
                            BatchNormConfig {
                                common: Common { dont_load, .. },
                                ..
                            },
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

            reader.read_f32_into::<LittleEndian>(biases.as_slice_mut().unwrap())?;
            reader.read_f32_into::<LittleEndian>(scales.as_slice_mut().unwrap())?;
            reader.read_f32_into::<LittleEndian>(rolling_mean.as_slice_mut().unwrap())?;
            reader.read_f32_into::<LittleEndian>(rolling_variance.as_slice_mut().unwrap())?;

            Ok(())
        }
    }

    impl ShortcutLayer {
        pub fn new(node: &ShortcutNode) -> Self {
            let ShortcutNode {
                config:
                    ShortcutConfig {
                        weights_type,
                        ref from,
                        ..
                    },
                output_shape: [_, _, out_c],
                ..
            } = *node;

            let out_c = out_c as usize;
            let num_input_layers = from.len() + 1;

            let weights = match weights_type {
                WeightsType::None => ShortcutWeights::None,
                WeightsType::PerFeature => ShortcutWeights::PerFeature(
                    Array1::from_shape_vec(num_input_layers, vec![0.0; num_input_layers]).unwrap(),
                ),
                WeightsType::PerChannel => ShortcutWeights::PerChannel(
                    Array2::from_shape_vec(
                        [num_input_layers, out_c],
                        vec![0.0; num_input_layers * out_c],
                    )
                    .unwrap(),
                ),
            };

            ShortcutLayer {
                node: node.clone(),
                weights,
            }
        }

        pub fn load_weights(&mut self, mut reader: impl ReadBytesExt) -> Result<()> {
            let Self {
                node:
                    ShortcutNode {
                        config:
                            ShortcutConfig {
                                common: Common { dont_load, .. },
                                ..
                            },
                        ..
                    },
                ref mut weights,
                ..
            } = *self;

            if dont_load {
                return Ok(());
            }

            match weights {
                ShortcutWeights::None => (),
                ShortcutWeights::PerFeature(weights) => {
                    reader.read_f32_into::<LittleEndian>(weights.as_slice_mut().unwrap())?;
                }
                ShortcutWeights::PerChannel(weights) => {
                    reader.read_f32_into::<LittleEndian>(weights.as_slice_mut().unwrap())?;
                }
            }

            Ok(())
        }
    }
}

mod weights {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct ScaleWeights {
        pub scales: Array1<f32>,
        pub rolling_mean: Array1<f32>,
        pub rolling_variance: Array1<f32>,
    }

    impl ScaleWeights {
        pub fn new(size: usize) -> Self {
            Self {
                scales: Array1::from_shape_vec(size, vec![0.0; size]).unwrap(),
                rolling_mean: Array1::from_shape_vec(size, vec![0.0; size]).unwrap(),
                rolling_variance: Array1::from_shape_vec(size, vec![0.0; size]).unwrap(),
            }
        }

        pub fn load_weights(&mut self, mut reader: impl ReadBytesExt) -> Result<()> {
            let Self {
                scales,
                rolling_mean,
                rolling_variance,
            } = self;

            reader.read_f32_into::<LittleEndian>(scales.as_slice_mut().unwrap())?;
            reader.read_f32_into::<LittleEndian>(rolling_mean.as_slice_mut().unwrap())?;
            reader.read_f32_into::<LittleEndian>(rolling_variance.as_slice_mut().unwrap())?;
            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    pub struct ConnectedWeights {
        pub biases: Array1<f32>,
        pub weights: Array2<f32>,
        pub scales: Option<ScaleWeights>,
    }

    #[derive(Debug, Clone)]
    pub enum ConvolutionalWeights {
        Owned {
            biases: Array1<f32>,
            weights: Array4<f32>,
            scales: Option<ScaleWeights>,
        },
        Ref {
            share_index: usize,
        },
    }

    #[derive(Debug, Clone)]
    pub struct BatchNormWeights {
        pub biases: Array1<f32>,
        pub scales: Array1<f32>,
        pub rolling_mean: Array1<f32>,
        pub rolling_variance: Array1<f32>,
    }

    #[derive(Debug, Clone)]
    pub enum ShortcutWeights {
        None,
        PerFeature(Array1<f32>),
        PerChannel(Array2<f32>),
    }
}
