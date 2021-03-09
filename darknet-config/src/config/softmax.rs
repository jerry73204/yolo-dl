use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "RawSoftmax")]
pub struct Softmax {
    pub groups: u64,
    pub temperature: R64,
    pub tree: Option<(PathBuf, Tree)>,
    pub spatial: R64,
    pub noloss: bool,
    pub common: Common,
}

impl TryFrom<RawSoftmax> for Softmax {
    type Error = Error;

    fn try_from(from: RawSoftmax) -> Result<Self, Self::Error> {
        let RawSoftmax {
            groups,
            temperature,
            tree_file,
            spatial,
            noloss,
            common,
        } = from;

        let tree = tree_file
            .map(|_path| -> Result<_> {
                unimplemented!();
            })
            .transpose()?;

        Ok(Self {
            groups,
            temperature,
            tree,
            spatial,
            noloss,
            common,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct RawSoftmax {
    #[serde(default = "defaults::softmax_groups")]
    pub groups: u64,
    #[serde(default = "defaults::temperature")]
    pub temperature: R64,
    pub tree_file: Option<PathBuf>,
    #[serde(default = "defaults::spatial")]
    pub spatial: R64,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub noloss: bool,
    #[serde(flatten)]
    pub common: Common,
}

impl TryFrom<Softmax> for RawSoftmax {
    type Error = Error;

    fn try_from(from: Softmax) -> Result<Self, Self::Error> {
        let Softmax {
            groups,
            temperature,
            tree,
            spatial,
            noloss,
            common,
        } = from;

        let tree_file = tree
            .map(|(_path, _tree)| -> Result<_> {
                unimplemented!();
            })
            .transpose()?;

        Ok(Self {
            groups,
            temperature,
            tree_file,
            spatial,
            noloss,
            common,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tree {}
