use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "RawDropout", into = "RawDropout")]
pub struct Dropout {
    pub probability: R64,
    pub dropblock: DropBlock,
    pub common: Common,
}

impl TryFrom<RawDropout> for Dropout {
    type Error = Error;

    fn try_from(from: RawDropout) -> Result<Self, Self::Error> {
        let RawDropout {
            probability,
            dropblock,
            dropblock_size_rel,
            dropblock_size_abs,
            common,
        } = from;

        let dropblock = match (dropblock, dropblock_size_rel, dropblock_size_abs) {
                (false, None, None) => DropBlock::None,
                (false, _, _) => bail!("neigher dropblock_size_rel nor dropblock_size_abs should be specified when dropblock is disabled"),
                (true, None, None) => bail!("dropblock is enabled, but none of dropblock_size_rel and dropblock_size_abs is specified"),
                (true, Some(val), None) => DropBlock::Relative(val),
                (true, None, Some(val)) => DropBlock::Absolute(val),
                (true, Some(_), Some(_)) => bail!("dropblock_size_rel and dropblock_size_abs cannot be specified together"),
            };

        Ok(Self {
            probability,
            dropblock,
            common,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct RawDropout {
    #[serde(default = "defaults::probability")]
    pub probability: R64,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub dropblock: bool,
    pub dropblock_size_rel: Option<R64>,
    pub dropblock_size_abs: Option<R64>,
    #[serde(flatten)]
    pub common: Common,
}

impl From<Dropout> for RawDropout {
    fn from(from: Dropout) -> Self {
        let Dropout {
            probability,
            dropblock,
            common,
        } = from;

        let (dropblock, dropblock_size_rel, dropblock_size_abs) = match dropblock {
            DropBlock::None => (false, None, None),
            DropBlock::Relative(val) => (false, Some(val), None),
            DropBlock::Absolute(val) => (false, None, Some(val)),
        };

        Self {
            probability,
            dropblock,
            dropblock_size_rel,
            dropblock_size_abs,
            common,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DropBlock {
    None,
    Absolute(R64),
    Relative(R64),
}
