use super::module::{GroupRef, Module, ModuleEx};
use crate::{common::*, utils};

pub use group_name::*;
mod group_name {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct GroupName(String);

    impl FromStr for GroupName {
        type Err = Error;

        fn from_str(name: &str) -> Result<Self, Self::Err> {
            ensure!(!name.is_empty(), "module name must not be empty");
            ensure!(!name.contains('.'), "module name must not contain dot '.'");
            Ok(Self(name.to_owned()))
        }
    }

    impl Serialize for GroupName {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            self.0.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for GroupName {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let text = String::deserialize(deserializer)?;
            let name = Self::from_str(&text)
                .map_err(|err| D::Error::custom(format!("invalid name: {:?}", err)))?;
            Ok(name)
        }
    }

    impl TryFrom<&str> for GroupName {
        type Error = Error;

        fn try_from(name: &str) -> Result<Self, Self::Error> {
            Self::from_str(name)
        }
    }

    impl Borrow<str> for GroupName {
        fn borrow(&self) -> &str {
            self.0.as_ref()
        }
    }

    impl AsRef<str> for GroupName {
        fn as_ref(&self) -> &str {
            self.0.as_ref()
        }
    }

    impl Display for GroupName {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            Display::fmt(&self.0, f)
        }
    }
}

pub use groups::*;
mod groups {
    use super::*;

    #[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
    #[derivative(PartialEq, Eq, Hash)]
    #[serde(try_from = "GroupsUnchecked", into = "GroupsUnchecked")]
    pub struct Groups(
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<GroupName, Group, _>"))]
        IndexMap<GroupName, Group>,
    );

    impl Groups {
        pub fn empty() -> Self {
            Self(IndexMap::new())
        }

        pub fn groups(&self) -> &IndexMap<GroupName, Group> {
            &self.0
        }

        pub fn load(path: impl AsRef<Path>) -> Result<Self> {
            Self::load_opt(path, 5)
        }

        pub fn load_opt(path: impl AsRef<Path>, max_depth: usize) -> Result<Self> {
            ensure!(max_depth > 0, "max_depth must be positive");
            let unit = GroupsUnit::load(path)?;
            Self::load_recursive(unit, max_depth, 0)
        }

        pub fn from_group_unit(group_unit: GroupsUnit) -> Result<Self> {
            Self::from_group_unit_opt(group_unit, 5)
        }

        pub fn from_group_unit_opt(group_unit: GroupsUnit, max_depth: usize) -> Result<Self> {
            Self::load_recursive(group_unit, max_depth, 0)
        }

        fn load_recursive(
            group_unit: GroupsUnit,
            max_depth: usize,
            curr_depth: usize,
        ) -> Result<Self> {
            let GroupsUnit { includes, groups } = group_unit;

            debug_assert!(max_depth > 0, "max_depth must be positive");
            ensure!(curr_depth < max_depth, "max_depth exceeded");

            // load included groups
            let included_groups_vec: Vec<_> = includes
                .into_iter()
                .map(|path| -> Result<_> {
                    let groups_unit = GroupsUnit::load(path)?;
                    let groups = Self::load_recursive(groups_unit, max_depth, curr_depth + 1)?;
                    Ok(groups)
                })
                .try_collect()?;
            let parent_groups = Self::merge(included_groups_vec)?;

            // check if group layer refer to valid groups
            Self::check_groups(&groups, Some(&parent_groups))?;

            // merge with included groups
            let merged_groups = Self::merge(vec![Groups(groups), parent_groups])?;

            Ok(merged_groups)
        }

        fn merge(groups_iter: impl IntoIterator<Item = Self>) -> Result<Self> {
            let groups: IndexMap<_, _> = groups_iter
                .into_iter()
                .flat_map(|Self(groups)| groups.into_iter())
                .try_fold(IndexMap::new(), |mut map, (name, group)| -> Result<_> {
                    let prev = map.insert(name.clone(), group);
                    ensure!(prev.is_none(), "duplicated group name '{}'", name);
                    Ok(map)
                })?;
            Ok(Self(groups))
        }

        fn check_groups(
            groups: &IndexMap<GroupName, Group>,
            included_groups: Option<&Groups>,
        ) -> Result<()> {
            groups
                .iter()
                .flat_map(|(group_name, group)| {
                    let Group(layers) = group;
                    layers.iter().map(move |layer| (group_name, layer))
                })
                .filter_map(|(group_name, layer)| match layer {
                    Module::GroupRef(group_ref) => Some((group_name, group_ref)),
                    _ => None,
                })
                .try_for_each(|(self_group_name, group_ref)| -> Result<_> {
                    let GroupRef {
                        group: ref_group_name,
                        ..
                    } = group_ref;

                    // forbit self-reference
                    ensure!(
                        self_group_name != ref_group_name,
                        "self group reference is not allowed"
                    );

                    // either the group is included or in the local group set
                    let group_from_included = included_groups
                        .map(|included_groups| included_groups.groups().get(ref_group_name))
                        .flatten();
                    let group_from_local = groups.get(ref_group_name);

                    match (group_from_included, group_from_local) {
                        (None, None) => bail!("undefined group reference '{}'", ref_group_name),
                        (Some(_), None) | (None, Some(_)) => (),
                        (Some(_), Some(_)) => bail!("duplicated group name '{}'", ref_group_name),
                    }

                    Ok(())
                })?;

            Ok(())
        }
    }

    #[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
    #[derivative(PartialEq, Eq, Hash)]
    pub struct GroupsUnchecked(
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<GroupName, Group, _>"))]
        IndexMap<GroupName, Group>,
    );

    impl TryFrom<GroupsUnchecked> for Groups {
        type Error = Error;

        fn try_from(GroupsUnchecked(groups): GroupsUnchecked) -> Result<Self, Self::Error> {
            Self::check_groups(&groups, None)?;
            Ok(Self(groups))
        }
    }

    impl From<Groups> for GroupsUnchecked {
        fn from(Groups(groups): Groups) -> Self {
            Self(groups)
        }
    }
}

pub use groups_unit::*;
mod groups_unit {
    use super::*;

    #[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
    #[derivative(PartialEq, Eq, Hash)]
    pub struct GroupsUnit {
        #[serde(default = "utils::empty_index_set::<PathBuf>")]
        #[derivative(Hash(hash_with = "utils::hash_vec_indexset::<PathBuf, _>"))]
        pub includes: IndexSet<PathBuf>,
        #[serde(default = "utils::empty_index_map::<GroupName, Group>")]
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<GroupName, Group, _>"))]
        pub groups: IndexMap<GroupName, Group>,
    }

    impl GroupsUnit {
        pub fn load(path: impl AsRef<Path>) -> Result<Self> {
            let path = path.as_ref();
            let config: Self = json5::from_str(
                &fs::read_to_string(path)
                    .with_context(|| format!("cannot open '{}'", path.display()))?,
            )
            .with_context(|| format!("failed to parse '{}'", path.display()))?;
            Ok(config)
        }
    }
}

pub use group_::*;
mod group_ {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(try_from = "GroupUnchecked", into = "GroupUnchecked")]
    pub struct Group(pub(super) Vec<Module>);

    impl Group {
        pub fn layers(&self) -> &[Module] {
            self.0.as_ref()
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    struct GroupUnchecked(Vec<Module>);

    impl TryFrom<GroupUnchecked> for Group {
        type Error = Error;

        fn try_from(GroupUnchecked(layers): GroupUnchecked) -> Result<Self, Self::Error> {
            layers.iter().filter_map(|layer| layer.name()).try_fold(
                HashSet::new(),
                |mut taken_names, name| {
                    let ok = taken_names.insert(name);
                    ensure!(ok, "duplicated name '{}'", name);
                    Ok(taken_names)
                },
            )?;
            Ok(Self(layers))
        }
    }

    impl From<Group> for GroupUnchecked {
        fn from(from: Group) -> Self {
            Self(from.0)
        }
    }
}
