use super::{Meta, Shape};
use crate::{common::*, utils};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "RawSoftmax")]
pub struct Softmax {
    pub groups: usize,
    pub temperature: R64,
    pub tree: Option<(PathBuf, Tree)>,
    pub spatial: R64,
    pub noloss: bool,
    pub common: Meta,
}

impl Softmax {
    pub fn output_shape(&self, input_shape: Shape) -> Shape {
        input_shape
    }
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
    #[serde(default = "utils::integer::<_, 1>")]
    pub groups: usize,
    #[serde(default = "utils::ratio::<_, 1, 1>")]
    pub temperature: R64,
    pub tree_file: Option<PathBuf>,
    #[serde(default = "utils::ratio::<_, 0, 1>")]
    pub spatial: R64,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub noloss: bool,
    #[serde(flatten)]
    pub common: Meta,
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
