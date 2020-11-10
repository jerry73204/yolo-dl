use crate::{
    common::*,
    config::{
        BatchNormConfig, CommonLayerOptions, ConnectedConfig, ConvolutionalConfig, DarknetConfig,
        LayerConfig, LayerConfigEx, LayerIndex, MaxPoolConfig, NetConfig, RouteConfig, Shape,
        ShortcutConfig, UpSampleConfig, YoloConfig,
    },
    shape::ShapeEx,
    weights::{BatchNormWeights, ConnectedWeights, ConvolutionalWeights, ShortcutWeights},
};

pub use layers::*;

#[derive(Debug)]
pub struct Model {
    layers: Vec<Layer>,
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
                    ref input_size,
                    batch,
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
                            vec![]
                        } else {
                            vec![layer_index - 1]
                        }
                    }
                    LayerConfig::Shortcut(conf) => conf
                        .from
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
                            Ok(index)
                        })
                        .try_collect()?,
                    LayerConfig::Route(conf) => conf
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
                            Ok(index)
                        })
                        .try_collect()?,
                };
                Ok((layer_index, from_indexes))
            })
            .try_collect()?;

        // topological sort
        let layers_map = {
            let graph = {
                let node_elements =
                    from_indexes_map
                        .keys()
                        .cloned()
                        .map(|layer_index| Element::Node {
                            weight: layer_index,
                        });
                let edge_elements =
                    from_indexes_map
                        .iter()
                        .flat_map(|(&layer_index, from_indexes)| {
                            from_indexes
                                .iter()
                                .cloned()
                                .map(move |from_index| Element::Edge {
                                    source: from_index,
                                    target: layer_index,
                                    weight: (),
                                })
                        });
                let graph =
                    DiGraphMap::<usize, ()>::from_elements(node_elements.chain(edge_elements));
                graph
            };

            let sorted_layer_indexes = petgraph::algo::toposort(&graph, None).map_err(|cycle| {
                format_err!("cycle detected at layer index {}", cycle.node_id())
            })?;

            let mut from_indexes_map = from_indexes_map;
            let layers_map: HashMap<_, _> = sorted_layer_indexes
                .into_iter()
                .map(|layer_index| {
                    let layer_config = &layers[layer_index];
                    let from_indexes = from_indexes_map.remove(&layer_index);
                    (layer_index, (layer_config, from_indexes))
                })
                .collect();

            debug_assert!(from_indexes_map.is_empty());
            layers_map
        };

        // compute shapes
        {
            #[derive(Debug)]
            struct State {
                shape: Shape,
            }

            let init_state = State {
                shape: input_size.clone(),
            };

            let final_state = layers_map.iter().try_fold(
                init_state,
                |mut state, (&layer_index, (layer_config, from_indexes))| -> Result<_> {
                    let CommonLayerOptions { dont_load, .. } = *layer_config.common();

                    if dont_load {
                        return Ok(state);
                    }

                    let layer: Layer = match layer_config {
                        LayerConfig::Convolutional(convolutional) => {
                            let input_shape = state
                                .shape
                                .hwc()
                                .ok_or_else(|| format_err!("invalid shape"))?;
                            Self::load_convolutional(convolutional, input_shape)?.into()
                        }
                        LayerConfig::Connected(connected) => {
                            let input_shape = state
                                .shape
                                .flat()
                                .ok_or_else(|| format_err!("invalid shape"))?;
                            Self::load_connected(connected, input_shape)?.into()
                        }
                        LayerConfig::BatchNorm(batch_norm) => {
                            let input_shape = state
                                .shape
                                .hwc()
                                .ok_or_else(|| format_err!("invalid shape"))?;
                            Self::load_batch_norm(batch_norm, input_shape)?.into()
                        }
                        LayerConfig::Shortcut(shortcut) => {
                            // Self::load_shortcut(shortcut, input_shape_list)?.into()
                            todo!()
                        }
                        LayerConfig::MaxPool(_) => {
                            let input_shape = state
                                .shape
                                .hwc()
                                .ok_or_else(|| format_err!("invalid shape"))?;
                            todo!();
                        }
                        LayerConfig::Route(_) => todo!(),
                        LayerConfig::UpSample(_) => {
                            let input_shape = state
                                .shape
                                .hwc()
                                .ok_or_else(|| format_err!("invalid shape"))?;
                            todo!();
                        }
                        LayerConfig::Yolo(_) => {
                            todo!();
                        }
                    };

                    #[cfg(debug_assertions)]
                    {
                        let kind = match layer {
                            Layer::Convolutional(_) => "convolutional",
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
                    }

                    Ok(state)
                },
            )?;
        }

        todo!();
    }

    pub fn load_weights<P>(&mut self, weights_file: P)
    where
        P: AsRef<Path>,
    {
    }
}

impl Model {
    fn load_convolutional(
        config: &ConvolutionalConfig,
        input_shape: [usize; 3],
    ) -> Result<ConvolutionalLayer> {
        let ConvolutionalConfig {
            filters,
            batch_normalize,
            flipped,
            padding,
            size,
            stride_x,
            stride_y,
            share_index,
            common: CommonLayerOptions {
                dont_load_scales, ..
            },
            ..
        } = *config;

        let output_shape = {
            let [in_h, in_w, in_c] = input_shape;
            let out_h = (in_h + 2 * padding - size) / stride_y + 1;
            let out_w = (in_w + 2 * padding - size) / stride_x + 1;
            [out_h, out_w, filters]
        };
        let weights = ConvolutionalWeights::default();

        Ok(ConvolutionalLayer {
            config: config.to_owned(),
            input_shape,
            output_shape,
            weights,
        })
    }

    fn load_connected(layer: &ConnectedConfig, input_shape: usize) -> Result<ConnectedLayer> {
        let ConnectedConfig {
            output,
            batch_normalize,
            common: CommonLayerOptions {
                dont_load_scales, ..
            },
            ..
        } = *layer;

        let output_shape = output;

        todo!();
    }

    fn load_batch_norm(
        _layer: &BatchNormConfig,
        input_shape: [usize; 3],
    ) -> Result<BatchNormLayer> {
        let output_shape = input_shape;
        todo!();
    }

    fn load_shortcut(layer: &ShortcutConfig, input_shapes: &[[usize; 3]]) -> Result<ShortcutLayer> {
        todo!();
    }
}

mod layers {
    pub use super::*;

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
        pub fn input_shape(&self) -> Shape {
            match self {
                Self::Connected(layer) => Shape::Flat(layer.input_shape()),
                Self::Convolutional(layer) => Shape::Hwc(layer.input_shape()),
                Self::Route(layer) => Shape::Hwc(layer.input_shape()),
                Self::Shortcut(layer) => Shape::Hwc(layer.input_shape()),
                Self::MaxPool(layer) => Shape::Hwc(layer.input_shape()),
                Self::UpSample(layer) => Shape::Hwc(layer.input_shape()),
                Self::Yolo(layer) => Shape::Hwc(layer.input_shape()),
                Self::BatchNorm(layer) => Shape::Hwc(layer.input_shape()),
            }
        }

        pub fn output_shape(&self) -> Shape {
            match self {
                Self::Connected(layer) => Shape::Flat(layer.output_shape()),
                Self::Convolutional(layer) => Shape::Hwc(layer.output_shape()),
                Self::Route(layer) => Shape::Hwc(layer.output_shape()),
                Self::Shortcut(layer) => Shape::Hwc(layer.output_shape()),
                Self::MaxPool(layer) => Shape::Hwc(layer.output_shape()),
                Self::UpSample(layer) => Shape::Hwc(layer.output_shape()),
                Self::Yolo(layer) => Shape::Hwc(layer.output_shape()),
                Self::BatchNorm(layer) => Shape::Hwc(layer.output_shape()),
            }
        }
    }

    macro_rules! declare_layer_type {
        ($name:ident, $config:ty, $weights:ty, $input_shape:ty, $output_shape:ty) => {
            #[derive(Debug)]
            pub struct $name {
                pub config: $config,
                pub input_shape: $input_shape,
                pub output_shape: $output_shape,
                pub weights: $weights,
            }

            impl $name {
                pub fn config(&self) -> &$config {
                    &self.config
                }

                pub fn weights(&self) -> &$weights {
                    &self.weights
                }

                pub fn input_shape(&self) -> $input_shape {
                    self.input_shape
                }

                pub fn output_shape(&self) -> $output_shape {
                    self.output_shape
                }
            }
        };
    }

    declare_layer_type!(
        ConnectedLayer,
        ConnectedConfig,
        ConnectedWeights,
        usize,
        usize
    );
    declare_layer_type!(
        ConvolutionalLayer,
        ConvolutionalConfig,
        ConvolutionalWeights,
        [usize; 3],
        [usize; 3]
    );
    declare_layer_type!(RouteLayer, RouteConfig, (), [usize; 3], [usize; 3]);
    declare_layer_type!(
        ShortcutLayer,
        ShortcutConfig,
        ShortcutWeights,
        [usize; 3],
        [usize; 3]
    );
    declare_layer_type!(MaxPoolLayer, MaxPoolConfig, (), [usize; 3], [usize; 3]);
    declare_layer_type!(UpSampleLayer, UpSampleConfig, (), [usize; 3], [usize; 3]);
    declare_layer_type!(YoloLayer, YoloConfig, (), [usize; 3], [usize; 3]);
    declare_layer_type!(
        BatchNormLayer,
        BatchNormConfig,
        BatchNormWeights,
        [usize; 3],
        [usize; 3]
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
        dbg!(&config);
        let model = Model::from_config(&config)?;
        Ok(())
    }
}
