use super::model::{GroupName, Model};
use crate::{common::*, utils};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Config {
    pub depth_multiple: Option<R64>,
    pub width_multiple: Option<R64>,
    pub include: Option<Vec<PathBuf>>,
    pub model: Model,
}

impl Config {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let config: Self = json5::from_str(
            &fs::read_to_string(path)
                .with_context(|| format!("cannot open '{}'", path.display()))?,
        )
        .with_context(|| format!("failed to parse '{}'", path.display()))?;
        Ok(config)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
#[derivative(Hash)]
pub struct ModelGroupsConfig {
    #[serde(default = "utils::empty_vec::<PathBuf>")]
    pub includes: Vec<PathBuf>,
    #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<GroupName, Model, _>"))]
    pub groups: IndexMap<GroupName, Model>,
}

impl ModelGroupsConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let config: Self = json5::from_str(
            &fs::read_to_string(path)
                .with_context(|| format!("cannot open '{}'", path.display()))?,
        )
        .with_context(|| format!("failed to parse '{}'", path.display()))?;
        Ok(config)
    }
}
