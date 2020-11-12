use crate::{
    common::*,
    config::{
        BatchNormConfig, CommonLayerOptions, ConnectedConfig, ConvolutionalConfig, DarknetConfig,
        ShortcutConfig,
    },
    model::{
        BatchNormLayerBase, ConnectedLayerBase, ConvolutionalLayerBase, LayerBase,
        MaxPoolLayerBase, ModelBase, RouteLayerBase, ShortcutLayerBase, UpSampleLayerBase,
        YoloLayerBase,
    },
};

pub use layer::*;
pub use model::*;
pub use weights::*;

mod model {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct DarknetModel {
        pub base: ModelBase,
        pub layers: IndexMap<usize, Layer>,
    }

    impl DarknetModel {
        pub fn new(model_base: &ModelBase) -> Result<Self> {
            // aggregate all computed features
            let layers: IndexMap<_, _> = {
                model_base
                    .layers
                    .iter()
                    .map(|(&layer_index, layer_base)| -> Result<_> {
                        let layer_base = layer_base.clone();

                        let layer = match layer_base {
                            LayerBase::Connected(conf) => {
                                let ConnectedLayerBase {
                                    config:
                                        ConnectedConfig {
                                            batch_normalize, ..
                                        },
                                    input_shape,
                                    output_shape,
                                    ..
                                } = conf;
                                let input_shape = input_shape as usize;

                                let weights = ConnectedWeights {
                                    biases: vec![0.0; input_shape],
                                    weights: vec![0.0; input_shape * output_shape as usize],
                                    scales: if batch_normalize {
                                        Some(ScaleWeights::new(output_shape))
                                    } else {
                                        None
                                    },
                                };

                                Layer::Connected(ConnectedLayer {
                                    base: conf,
                                    weights,
                                })
                            }
                            LayerBase::Convolutional(conf) => {
                                let ConvolutionalLayerBase {
                                    config:
                                        ConvolutionalConfig {
                                            ref share_index,
                                            filters,
                                            batch_normalize,
                                            ..
                                        },
                                    input_shape: [_, _, in_c],
                                    ..
                                } = conf;

                                let weights = if let Some(share_index) = share_index {
                                    let share_index = share_index
                                        .to_absolute(layer_index)
                                        .ok_or_else(|| format_err!("invalid layer index"))?;
                                    ConvolutionalWeights::Ref { share_index }
                                } else {
                                    let weights = vec![0.0; conf.config.num_weights(in_c)?];
                                    let biases = vec![0.0; filters as usize];
                                    let scales = if batch_normalize {
                                        Some(ScaleWeights::new(filters))
                                    } else {
                                        None
                                    };

                                    ConvolutionalWeights::Owned {
                                        biases,
                                        weights,
                                        scales,
                                    }
                                };

                                Layer::Convolutional(ConvolutionalLayer {
                                    base: conf,
                                    weights,
                                })
                            }
                            LayerBase::Route(conf) => Layer::Route(RouteLayer { base: conf }),
                            LayerBase::Shortcut(conf) => {
                                let [_, _, out_c] = conf.output_shape;
                                let weights = ShortcutWeights {
                                    weights: conf
                                        .config
                                        .num_weights(out_c)
                                        .map(|num_weights| vec![0.0; num_weights]),
                                };

                                Layer::Shortcut(ShortcutLayer {
                                    base: conf,
                                    weights,
                                })
                            }
                            LayerBase::MaxPool(conf) => Layer::MaxPool(MaxPoolLayer { base: conf }),
                            LayerBase::UpSample(conf) => {
                                Layer::UpSample(UpSampleLayer { base: conf })
                            }
                            LayerBase::BatchNorm(conf) => {
                                let [_h, _w, channels] = conf.inout_shape;
                                let channels = channels as usize;

                                let biases = vec![0.0; channels];
                                let scales = vec![0.0; channels];
                                let rolling_mean = vec![0.0; channels];
                                let rolling_variance = vec![0.0; channels];

                                let weights = BatchNormWeights {
                                    biases,
                                    scales,
                                    rolling_mean,
                                    rolling_variance,
                                };

                                Layer::BatchNorm(BatchNormLayer {
                                    base: conf,
                                    weights,
                                })
                            }
                            LayerBase::Yolo(conf) => Layer::Yolo(YoloLayer { base: conf }),
                        };

                        Ok((layer_index, layer))
                    })
                    .try_collect()?
            };

            Ok(Self {
                base: model_base.clone(),
                layers,
            })
        }

        pub fn from_config_file<P>(config_file: P) -> Result<Self>
        where
            P: AsRef<Path>,
        {
            let config = DarknetConfig::load(config_file)?;
            let base = ModelBase::from_config(&config)?;
            Self::new(&base)
        }

        pub fn from_config(config: &DarknetConfig) -> Result<Self> {
            let base = ModelBase::from_config(config)?;
            Self::new(&base)
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
            self.base.seen = seen;
            self.base.cur_iteration = self.base.net.iteration(seen);

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
        ($name:ident, $base:ty, $weights:ty) => {
            #[derive(Debug, Clone)]
            pub struct $name {
                pub base: $base,
                pub weights: $weights,
            }
        };
        ($name:ident, $base:ty) => {
            #[derive(Debug, Clone)]
            pub struct $name {
                pub base: $base,
            }
        };
    }

    #[derive(Debug, Clone)]
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
    }

    declare_darknet_layer!(ConnectedLayer, ConnectedLayerBase, ConnectedWeights);
    declare_darknet_layer!(
        ConvolutionalLayer,
        ConvolutionalLayerBase,
        ConvolutionalWeights
    );
    declare_darknet_layer!(BatchNormLayer, BatchNormLayerBase, BatchNormWeights);
    declare_darknet_layer!(ShortcutLayer, ShortcutLayerBase, ShortcutWeights);
    declare_darknet_layer!(RouteLayer, RouteLayerBase);
    declare_darknet_layer!(MaxPoolLayer, MaxPoolLayerBase);
    declare_darknet_layer!(UpSampleLayer, UpSampleLayerBase);
    declare_darknet_layer!(YoloLayer, YoloLayerBase);

    impl ConnectedLayer {
        pub fn load_weights(
            &mut self,
            mut reader: impl ReadBytesExt,
            transpose: bool,
        ) -> Result<()> {
            let Self {
                base:
                    ConnectedLayerBase {
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
                base:
                    ConvolutionalLayerBase {
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
                base:
                    BatchNormLayerBase {
                        config:
                            BatchNormConfig {
                                common: CommonLayerOptions { dont_load, .. },
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
                base:
                    ShortcutLayerBase {
                        config:
                            ShortcutConfig {
                                common: CommonLayerOptions { dont_load, .. },
                                ..
                            },
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

mod weights {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct ScaleWeights {
        pub scales: Vec<f32>,
        pub rolling_mean: Vec<f32>,
        pub rolling_variance: Vec<f32>,
    }

    impl ScaleWeights {
        pub fn new(size: u64) -> Self {
            let size = size as usize;
            Self {
                scales: vec![0.0; size],
                rolling_mean: vec![0.0; size],
                rolling_variance: vec![0.0; size],
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct ConnectedWeights {
        pub biases: Vec<f32>,
        pub weights: Vec<f32>,
        pub scales: Option<ScaleWeights>,
    }

    #[derive(Debug, Clone)]
    pub enum ConvolutionalWeights {
        Owned {
            biases: Vec<f32>,
            weights: Vec<f32>,
            scales: Option<ScaleWeights>,
        },
        Ref {
            share_index: usize,
        },
    }

    #[derive(Debug, Clone)]
    pub struct BatchNormWeights {
        pub biases: Vec<f32>,
        pub scales: Vec<f32>,
        pub rolling_mean: Vec<f32>,
        pub rolling_variance: Vec<f32>,
    }

    #[derive(Debug, Clone)]
    pub struct ShortcutWeights {
        pub weights: Option<Vec<f32>>,
    }
}
