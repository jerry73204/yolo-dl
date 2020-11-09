use crate::{
    common::*,
    config::{CommonLayerOptions, Config, Connected, Convolutional, Layer, Net, Shape},
};

pub fn load<P1, P2>(config_file: P1, weights_file: P2) -> Result<()>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
{
    // load config file
    let Config {
        net: Net {
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
            pub fn read<T>(&mut self, size: usize) -> Result<OwningRef<Arc<Vec<u8>>, [T]>> {
                todo!();
            }
        }

        // loop
        let init_state = State {
            cursor: Cursor { buffer },
            shape: input_size,
        };

        layers
            .iter()
            .try_fold(init_state, |mut state, layer| -> Result<_> {
                let State { cursor, shape } = &mut state;

                let weights: Weights = match layer {
                    Layer::Connected(connected) => {
                        let Connected {
                            output,
                            batch_normalize,
                            common:
                                CommonLayerOptions {
                                    dont_load_scales, ..
                                },
                            ..
                        } = *connected;

                        let inputs = match *shape {
                            Shape::Flat(inputs) => inputs,
                            _ => bail!("invalid shape"),
                        };

                        let (biases, weights) = {
                            let biases = cursor.read::<R32>(output)?;
                            let weights = cursor.read::<R32>(inputs * output)?;
                            if transpose {
                                // TODO
                                (biases, weights)
                            } else {
                                (biases, weights)
                            }
                        };

                        let scales = if batch_normalize && !dont_load_scales {
                            let scales = cursor.read::<R32>(output)?;
                            let rolling_mean = cursor.read::<R32>(output)?;
                            let rolling_variance = cursor.read::<R32>(output)?;

                            Some(ConnectedScaleWeights {
                                scales: Box::new(scales),
                                rolling_mean: Box::new(rolling_mean),
                                rolling_variance: Box::new(rolling_variance),
                            })
                        } else {
                            None
                        };

                        ConnectedWeights {
                            biases: Box::new(biases),
                            weights: Box::new(weights),
                            scales,
                        }
                        .into()
                    }
                    Layer::BatchNorm(batchnorm) => {
                        let channels = match *shape {
                            Shape::Hwc([_, _, channels]) => channels,
                            _ => bail!("invalid input shape"),
                        };
                        let biases = cursor.read::<R32>(channels)?;
                        let scales = cursor.read::<R32>(channels)?;
                        let rolling_mean = cursor.read::<R32>(channels)?;
                        let rolling_variance = cursor.read::<R32>(channels)?;
                        BatchNormWeights {
                            biases: Box::new(biases),
                            scales: Box::new(scales),
                            rolling_mean: Box::new(rolling_mean),
                            rolling_variance: Box::new(rolling_variance),
                        }
                        .into()
                    }
                    _ => todo!(),
                };

                Ok(state)
            })?;
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

#[derive(Debug, Clone, BinRead)]
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

#[derive(Debug)]
pub struct ConnectedWeights {
    pub biases: Box<dyn Buffer<R32>>,
    pub weights: Box<dyn Buffer<R32>>,
    pub scales: Option<ConnectedScaleWeights>,
}

#[derive(Debug)]
pub struct ConnectedScaleWeights {
    pub scales: Box<dyn Buffer<R32>>,
    pub rolling_mean: Box<dyn Buffer<R32>>,
    pub rolling_variance: Box<dyn Buffer<R32>>,
}

#[derive(Debug)]
pub struct ConvolutionalWeights {} // TODO

#[derive(Debug)]
pub struct BatchNormWeights {
    pub biases: Box<dyn Buffer<R32>>,
    pub scales: Box<dyn Buffer<R32>>,
    pub rolling_mean: Box<dyn Buffer<R32>>,
    pub rolling_variance: Box<dyn Buffer<R32>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wtf() -> Result<()> {
        let config_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("yolov4.cfg");
        let weights_path = "/home/jerry73204/Downloads/yolov4.weights";
        load(config_path, weights_path)?;
        Ok(())
    }
}
