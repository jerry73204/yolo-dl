use super::group::{Group, GroupName, Groups, GroupsUnit};
use crate::{common::*, utils};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Model {
    pub groups: Groups,
    pub main_group: GroupName,
}

impl Model {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_opt(path, 5)
    }

    pub fn load_opt(path: impl AsRef<Path>, max_depth: usize) -> Result<Self> {
        let path = path.as_ref();

        // load unchecked config
        let ModelUnchecked {
            includes,
            groups: inplace_groups,
            main_group,
        } = json5::from_str(
            &fs::read_to_string(path)
                .with_context(|| format!("cannot open '{}'", path.display()))?,
        )
        .with_context(|| format!("failed to parse '{}'", path.display()))?;

        let group_unit = GroupsUnit {
            includes,
            groups: inplace_groups,
        };
        let groups = Groups::from_group_unit_opt(group_unit, max_depth)?;

        ensure!(
            groups.groups().contains_key(&main_group),
            "the group '{}' does not exist",
            main_group
        );

        Ok(Self { groups, main_group })
    }
}

#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(PartialEq, Eq, Hash)]
struct ModelUnchecked {
    #[serde(default = "utils::empty_index_set::<PathBuf>")]
    #[derivative(Hash(hash_with = "utils::hash_vec_indexset::<PathBuf, _>"))]
    pub includes: IndexSet<PathBuf>,
    #[serde(default = "utils::empty_index_map::<GroupName, Group>")]
    #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<GroupName, Group, _>"))]
    pub groups: IndexMap<GroupName, Group>,
    pub main_group: GroupName,
}
