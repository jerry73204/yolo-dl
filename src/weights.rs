use crate::{
    common::*,
    config::{
        BatchNormConfig, CommonLayerOptions, ConnectedConfig, ConvolutionalConfig, DarknetConfig,
        LayerConfig, LayerConfigEx, NetConfig, Shape, ShortcutConfig,
    },
};

pub fn load<P1, P2>(config_file: P1, weights_file: P2) -> Result<()>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    // load config file
    let DarknetConfig {
        net: NetConfig {
            input_size, batch, ..
        },
        layers,
    } = serde_ini::from_str(&fs::read_to_string(config_file)?)?;

    // load weights file
    let (version, transpose, buffer) = move || -> Result<_, binread::Error> {
        let mut reader = BufReader::new(File::open(weights_file)?);

        let version: Version = reader.read_le()?;
        let Version { major, minor, .. } = version;

        let seen: u64 = if major * 10 + minor >= 2 {
            reader.read_le()?
        } else {
            let seen: u32 = reader.read_le()?;
            seen as u64
        };
        let transpose = (major > 1000) || (minor > 1000);

        let buffer = {
            let mut buffer = vec![];
            reader.read_to_end(&mut buffer)?;
            ArcRef::new(Arc::new(buffer)).map(|buf| buf.as_slice())
        };
        Ok((version, transpose, buffer))
    }()
    .map_err(|err| format_err!("failed to parse weight file: {:?}", err))?;

    // construct layers
    {
        // types
        #[derive(Debug)]
        struct State {
            cursor: Cursor,
            shape: Shape,
        }

        #[derive(Debug)]
        struct Cursor {
            buffer: OwningRef<Arc<Vec<u8>>, [u8]>,
        }

        impl Cursor {
            pub fn read<T>(&mut self, numel: usize) -> Result<OwningRef<Arc<Vec<u8>>, [T]>> {
                let elem_size = mem::size_of::<T>();
                let size = elem_size * numel;

                ensure!(size <= self.buffer.len(), "early EOF");
                let taken = self.buffer.clone().map(|buf| unsafe {
                    slice::from_raw_parts(buf.split_at(size).0.as_ptr() as *const T, numel)
                });
                self.buffer = self.buffer.clone().map(|buf| buf.split_at(size).1);

                Ok(taken)
            }

            pub fn remaining_len(&self) -> usize {
                self.buffer.len()
            }

            pub fn is_eof(&self) -> bool {
                self.buffer.len() == 0
            }
        }

        // layer loading functions

        let load_convolutional = |layer: &ConvolutionalConfig, state: &mut State| -> Result<_> {
            let State { cursor, shape } = state;
            let ConvolutionalConfig {
                filters,
                batch_normalize,
                flipped,
                padding,
                size,
                stride_x,
                stride_y,
                share_index,
                common:
                    CommonLayerOptions {
                        dont_load_scales, ..
                    },
                ..
            } = *layer;

            let [height, width, channels] = match *shape {
                Shape::Hwc([h, w, c]) => [h, w, c],
                _ => bail!("invalid input shape"),
            };

            let weights = if let Some(share_index) = share_index {
                warn!("TODO: implement shared_index");
                None
            } else {
                let biases = cursor.read::<R32>(filters)?;
                let scales = if batch_normalize && !dont_load_scales {
                    let scales = cursor.read::<R32>(filters)?;
                    let rolling_mean = cursor.read::<R32>(filters)?;
                    let rolling_variance = cursor.read::<R32>(filters)?;

                    Some(ScaleWeights {
                        scales: Box::new(scales),
                        rolling_mean: Box::new(rolling_mean),
                        rolling_variance: Box::new(rolling_variance),
                    })
                } else {
                    None
                };
                let weights = {
                    let num_weights = layer.num_weights(channels)?;
                    let weights = cursor.read::<R32>(num_weights)?;

                    if flipped {
                        warn!("TODO: implement flipped option");
                    }

                    weights
                };

                Some(ConvolutionalWeights {
                    biases: Box::new(biases),
                    weights: Box::new(weights),
                    scales,
                })
            };

            // update shape
            *shape = {
                let new_height = (height + 2 * padding - size) / stride_y + 1;
                let new_width = (width + 2 * padding - size) / stride_x + 1;
                Shape::Hwc([new_height, new_width, filters])
            };

            Ok(weights)
        };

        let load_connected = |layer: &ConnectedConfig, state: &mut State| -> Result<_> {
            let State { cursor, shape } = state;
            let ConnectedConfig {
                output,
                batch_normalize,
                common:
                    CommonLayerOptions {
                        dont_load_scales, ..
                    },
                ..
            } = *layer;

            let inputs = match *shape {
                Shape::Flat(inputs) => inputs,
                _ => bail!("invalid shape"),
            };

            let (biases, weights) = {
                let biases = cursor.read::<R32>(output)?;
                let weights = cursor.read::<R32>(inputs * output)?;
                if transpose {
                    warn!("TODO: transpose");
                    (biases, weights)
                } else {
                    (biases, weights)
                }
            };

            let scales = if batch_normalize && !dont_load_scales {
                let scales = cursor.read::<R32>(output)?;
                let rolling_mean = cursor.read::<R32>(output)?;
                let rolling_variance = cursor.read::<R32>(output)?;

                Some(ScaleWeights {
                    scales: Box::new(scales),
                    rolling_mean: Box::new(rolling_mean),
                    rolling_variance: Box::new(rolling_variance),
                })
            } else {
                None
            };

            // update shape
            *shape = Shape::Flat(output);

            Ok(ConnectedWeights {
                biases: Box::new(biases),
                weights: Box::new(weights),
                scales,
            })
        };

        let load_batch_norm = |_layer: &BatchNormConfig, state: &mut State| -> Result<_> {
            let State { cursor, shape } = state;
            let channels = match *shape {
                Shape::Hwc([_, _, channels]) => channels,
                _ => bail!("invalid input shape"),
            };
            let biases = cursor.read::<R32>(channels)?;
            let scales = cursor.read::<R32>(channels)?;
            let rolling_mean = cursor.read::<R32>(channels)?;
            let rolling_variance = cursor.read::<R32>(channels)?;

            Ok(BatchNormWeights {
                biases: Box::new(biases),
                scales: Box::new(scales),
                rolling_mean: Box::new(rolling_mean),
                rolling_variance: Box::new(rolling_variance),
            })
        };

        let load_shortcut = |layer: &ShortcutConfig, state: &mut State| -> Result<_> {
            let State { cursor, shape } = state;
            let channels = match *shape {
                Shape::Hwc([_, _, channels]) => channels,
                _ => bail!("invalid input shape"),
            };
            let weights = {
                let num_weights = layer.num_weights(channels);
                cursor.read::<R32>(num_weights)?
            };

            Ok(ShortcutWeights {
                weights: Box::new(weights),
            })
        };

        // loop
        let init_state = State {
            cursor: Cursor { buffer },
            shape: input_size,
        };
        let final_state = layers.iter().enumerate().try_fold(
            init_state,
            |mut state, (layer_index, layer)| -> Result<_> {
                let CommonLayerOptions { dont_load, .. } = *layer.common();

                if dont_load {
                    return Ok(state);
                }

                let input_shape = state.shape.clone();

                let weights: Option<Weights> = match layer {
                    LayerConfig::Convolutional(convolutional) => {
                        load_convolutional(convolutional, &mut state)?.map(Into::into)
                    }
                    LayerConfig::Connected(connected) => {
                        Some(load_connected(connected, &mut state)?.into())
                    }
                    LayerConfig::BatchNorm(batch_norm) => {
                        Some(load_batch_norm(batch_norm, &mut state)?.into())
                    }
                    LayerConfig::Shortcut(shortcut) => {
                        Some(load_shortcut(shortcut, &mut state)?.into())
                    }
                    LayerConfig::MaxPool(_)
                    | LayerConfig::Route(_)
                    | LayerConfig::UpSample(_)
                    | LayerConfig::Yolo(_) => None,
                };

                #[cfg(debug_assertions)]
                {
                    let output_shape = state.shape.clone();
                    let kind = match layer {
                        LayerConfig::Convolutional(_) => "convolutional",
                        LayerConfig::Connected(_) => "connected",
                        LayerConfig::BatchNorm(_) => "batch_norm",
                        LayerConfig::Shortcut(_) => "shortcut",
                        LayerConfig::MaxPool(_) => "max_pool",
                        LayerConfig::Route(_) => "route",
                        LayerConfig::UpSample(_) => "up_sample",
                        LayerConfig::Yolo(_) => "yolo",
                    };
                    debug!(
                        "{}\t{}\t{:?}\t{:?}",
                        layer_index, kind, input_shape, output_shape
                    );
                }

                Ok(state)
            },
        )?;

        ensure!(
            final_state.cursor.is_eof(),
            "the weights file is not totally consumed"
        )
    }

    Ok(())
}

pub trait Buffer<T>
where
    Self: AsRef<[T]> + Debug,
{
}

impl<T, Owned> Buffer<T> for OwningRef<Owned, [T]>
where
    T: Debug,
    Owned: Debug,
{
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, BinRead)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub revision: u32,
}

// weight types

#[derive(Debug)]
pub enum Weights {
    Connected(ConnectedWeights),
    Convolutional(ConvolutionalWeights),
    BatchNorm(BatchNormWeights),
    Shortcut(ShortcutWeights),
}

impl From<ConnectedWeights> for Weights {
    fn from(weights: ConnectedWeights) -> Self {
        Self::Connected(weights)
    }
}

impl From<ConvolutionalWeights> for Weights {
    fn from(weights: ConvolutionalWeights) -> Self {
        Self::Convolutional(weights)
    }
}

impl From<BatchNormWeights> for Weights {
    fn from(weights: BatchNormWeights) -> Self {
        Self::BatchNorm(weights)
    }
}

impl From<ShortcutWeights> for Weights {
    fn from(weights: ShortcutWeights) -> Self {
        Self::Shortcut(weights)
    }
}

#[derive(Debug)]
pub struct ScaleWeights {
    pub scales: Box<dyn Buffer<R32>>,
    pub rolling_mean: Box<dyn Buffer<R32>>,
    pub rolling_variance: Box<dyn Buffer<R32>>,
}

#[derive(Debug)]
pub struct ConnectedWeights {
    pub biases: Box<dyn Buffer<R32>>,
    pub weights: Box<dyn Buffer<R32>>,
    pub scales: Option<ScaleWeights>,
}

#[derive(Debug)]
pub struct ConvolutionalWeights {
    pub biases: Box<dyn Buffer<R32>>,
    pub weights: Box<dyn Buffer<R32>>,
    pub scales: Option<ScaleWeights>,
}

impl Default for ConvolutionalWeights {
    fn default() -> Self {
        todo!();
    }
}

#[derive(Debug)]
pub struct BatchNormWeights {
    pub biases: Box<dyn Buffer<R32>>,
    pub scales: Box<dyn Buffer<R32>>,
    pub rolling_mean: Box<dyn Buffer<R32>>,
    pub rolling_variance: Box<dyn Buffer<R32>>,
}

#[derive(Debug)]
pub struct ShortcutWeights {
    pub weights: Box<dyn Buffer<R32>>,
}
