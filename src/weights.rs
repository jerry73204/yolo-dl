use crate::{
    common::*,
    config::{Config, Convolutional, Layer},
};

pub fn load<P>(config: &Config, weights_file: P) -> Result<()>
where
    P: AsRef<Path>,
{
    let mut reader = BufReader::new(File::open(weights_file)?);
    let mut parse = move || -> Result<_, binread::Error> {
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
            ArcRef::new(Arc::new(buffer))
        };

        config
            .layers
            .iter()
            .try_fold(buffer, |buffer, layer| -> Result<_, binread::Error> {
                let buffer = buffer.clone();

                match layer {
                    Layer::Convolutional(..) => {}
                    _ => todo!(),
                }

                Ok(buffer)
            })?;

        Ok(())
    };

    parse().map_err(|err| format_err!("failed to parse weight file: {:?}", err))?;

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, BinRead)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub revision: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConnectedWeights {
    pub biases: OwningRef<Arc<Vec<u8>>, [R32]>,
    pub weights: OwningRef<Arc<Vec<u8>>, [R32]>,
    pub scales: Option<ConnectedWeightsScales>,
}

impl ConnectedWeights {
    pub fn new(layer: &Convolutional, buffer: ArcRef<Vec<u8>>) -> Self {
        todo!();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConnectedWeightsScales {
    pub scales: OwningRef<Arc<Vec<u8>>, [R32]>,
    pub rolling_mean: OwningRef<Arc<Vec<u8>>, [R32]>,
    pub rolling_variance: OwningRef<Arc<Vec<u8>>, [R32]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchNormWeights {
    pub biases: OwningRef<Arc<Vec<u8>>, [R32]>,
    pub scales: OwningRef<Arc<Vec<u8>>, [R32]>,
    pub rolling_mean: OwningRef<Arc<Vec<u8>>, [R32]>,
    pub rolling_variance: OwningRef<Arc<Vec<u8>>, [R32]>,
}
