use super::{
    config::ModelGroupsConfig,
    misc::Shape,
    module::{Group, Module, ModuleEx},
};
use crate::{common::*, utils};

pub use group_name::*;
pub use model::*;
pub use model_groups::*;
pub use module_ident::*;
pub use module_name::*;
pub use module_path::*;

mod module_name {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct ModuleName(String);

    impl ModuleName {
        pub fn new<'a>(name: impl Into<Cow<'a, str>>) -> Result<Self> {
            let name = name.into().into_owned();
            ensure!(!name.is_empty(), "module name must not be empty");
            ensure!(!name.contains('.'), "module name must not contain dot '.'");
            Ok(Self(name))
        }
    }

    impl Serialize for ModuleName {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            self.0.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for ModuleName {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let text = String::deserialize(deserializer)?;
            let name = Self::new(text)
                .map_err(|err| D::Error::custom(format!("invalid name: {:?}", err)))?;
            Ok(name)
        }
    }

    impl TryFrom<&str> for ModuleName {
        type Error = Error;

        fn try_from(name: &str) -> Result<Self, Self::Error> {
            Self::new(name)
        }
    }

    impl Borrow<str> for ModuleName {
        fn borrow(&self) -> &str {
            self.0.as_ref()
        }
    }

    impl AsRef<str> for ModuleName {
        fn as_ref(&self) -> &str {
            self.0.as_ref()
        }
    }

    impl Display for ModuleName {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            self.0.fmt(f)
        }
    }
}

mod group_name {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct GroupName(String);

    impl GroupName {
        pub fn new<'a>(name: impl Into<Cow<'a, str>>) -> Result<Self> {
            let name = name.into().into_owned();
            ensure!(!name.is_empty(), "module name must not be empty");
            ensure!(!name.contains('.'), "module name must not contain dot '.'");
            Ok(Self(name))
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
            let name = Self::new(text)
                .map_err(|err| D::Error::custom(format!("invalid name: {:?}", err)))?;
            Ok(name)
        }
    }

    impl TryFrom<&str> for GroupName {
        type Error = Error;

        fn try_from(name: &str) -> Result<Self, Self::Error> {
            Self::new(name)
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
            self.0.fmt(f)
        }
    }
}

mod module_ident {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum ModuleIdent {
        Unnamed,
        Named(ModuleName),
    }

    impl ModuleIdent {
        pub fn new<'a>(name: impl Into<Cow<'a, str>>) -> Result<Self> {
            let name = name.into();
            let name: &str = name.borrow();
            let ident = match name {
                "" => Self::Unnamed,
                _ => Self::Named(name.try_into()?),
            };
            Ok(ident)
        }

        pub fn name(&self) -> Option<&ModuleName> {
            match self {
                Self::Unnamed => None,
                Self::Named(name) => Some(name),
            }
        }
    }

    impl Serialize for ModuleIdent {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            match self {
                Self::Unnamed => "",
                Self::Named(name) => name.as_ref(),
            }
            .serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for ModuleIdent {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let text = String::deserialize(deserializer)?;
            let ident = Self::new(text)
                .map_err(|err| D::Error::custom(format!("invalid name: {:?}", err)))?;
            Ok(ident)
        }
    }

    impl TryFrom<&str> for ModuleIdent {
        type Error = Error;

        fn try_from(name: &str) -> Result<Self, Self::Error> {
            Self::new(name)
        }
    }

    impl From<ModuleName> for ModuleIdent {
        fn from(name: ModuleName) -> Self {
            ModuleIdent::Named(name)
        }
    }

    impl Display for ModuleIdent {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            match self {
                Self::Unnamed => "[unnamed]",
                Self::Named(name) => name.as_ref(),
            }
            .fmt(f)
        }
    }
}

mod module_path {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum ModulePath {
        Layer(ModuleName),
        GroupRef {
            layer: ModuleName,
            output: ModuleName,
        },
    }

    impl ModulePath {
        pub fn new<'a>(name: impl Into<Cow<'a, str>>) -> Result<Self> {
            let name = name.into();
            let name: &str = name.borrow();

            let mut tokens = name.split('.');
            let first = tokens.next();
            let second = tokens.next();
            let third = tokens.next();

            let path = match (first, second, third) {
                (Some(first), None, None) => {
                    ensure!(!first.is_empty(), "invalid layer path '{}'", name);
                    Self::Layer(first.try_into()?)
                }
                (Some(first), Some(second), None) => {
                    ensure!(
                        !first.is_empty() && !second.is_empty(),
                        "invalid layer path '{}'",
                        name
                    );
                    Self::GroupRef {
                        layer: first.try_into()?,
                        output: second.try_into()?,
                    }
                }
                _ => bail!("layer path must contain at most one dot symbol '.'"),
            };

            Ok(path)
        }
    }

    impl Serialize for ModulePath {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let text: Cow<'_, str> = match self {
                Self::Layer(name) => name.as_ref().into(),
                Self::GroupRef { layer, output } => {
                    format!("{}.{}", layer.as_ref(), output.as_ref()).into()
                }
            };
            text.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for ModulePath {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let text = String::deserialize(deserializer)?;
            let ident = Self::new(text)
                .map_err(|err| D::Error::custom(format!("invalid name: {:?}", err)))?;
            Ok(ident)
        }
    }

    impl TryFrom<&str> for ModulePath {
        type Error = Error;

        fn try_from(name: &str) -> Result<Self, Self::Error> {
            Self::new(name)
        }
    }

    impl Display for ModulePath {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            let text: Cow<'_, str> = match self {
                Self::Layer(name) => name.as_ref().into(),
                Self::GroupRef { layer, output } => {
                    format!("{}.{}", layer.as_ref(), output.as_ref()).into()
                }
            };
            text.fmt(f)
        }
    }
}

mod model_groups {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    pub struct ModelGroups(
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<GroupName, Model, _>"))]
        IndexMap<GroupName, Model>,
    );

    impl ModelGroups {
        pub fn load(path: impl AsRef<Path>) -> Result<Self> {
            Self::load_opt(path, 5)
        }

        pub fn load_opt(path: impl AsRef<Path>, max_depth: usize) -> Result<Self> {
            ensure!(max_depth > 0, "max_depth must be positive");
            Self::load_recursive(path, max_depth, 0)
        }

        fn load_recursive(
            path: impl AsRef<Path>,
            max_depth: usize,
            curr_depth: usize,
        ) -> Result<Self> {
            debug_assert!(max_depth > 0, "max_depth must be positive");
            ensure!(curr_depth < max_depth, "max_depth exceeded");

            let ModelGroupsConfig {
                includes,
                groups: local_groups,
            } = ModelGroupsConfig::from_path(path)?;

            // load included groups
            let included_groups_vec: Vec<_> = includes
                .into_iter()
                .map(|path| Self::load_recursive(path, max_depth, curr_depth + 1))
                .try_collect()?;
            let parent_groups = Self::merge(included_groups_vec)?;

            // check if group layer refer to valid groups
            let ref_graph = local_groups
                .iter()
                .flat_map(|(curr_group, model)| {
                    let Model { layers, .. } = model;
                    layers
                        .iter()
                        .map(move |(_layer_name, layer)| (curr_group, layer))
                })
                .filter_map(|(curr_group, layer)| match layer {
                    Module::Group(Group {
                        group: tgt_group, ..
                    }) => Some((curr_group, tgt_group)),
                    _ => None,
                })
                .try_fold(
                    DiGraphMap::new(),
                    |mut graph, (curr_group, tgt_group)| -> Result<_> {
                        ensure!(
                            curr_group != tgt_group,
                            "the Group layer must not refer to the self group '{}'",
                            tgt_group
                        );

                        let Self(parent_groups) = &parent_groups;

                        if parent_groups.contains_key(tgt_group) {
                            // parent groups
                            Ok(graph)
                        } else if local_groups.contains_key(tgt_group) {
                            // local group
                            graph.add_edge(curr_group, tgt_group, ());
                            Ok(graph)
                        } else {
                            bail!("'{}' does not refer to a valid group", tgt_group);
                        }
                    },
                )?;

            // check cyclic group references in local groups
            petgraph::algo::toposort(&ref_graph, None).map_err(|cycle| {
                format_err!(
                    "cyclic group reference foudn at '{}' group",
                    cycle.node_id()
                )
            })?;

            // check every layer refers to a valid input layer
            local_groups
                .iter()
                .try_for_each(|(self_group_name, model)| -> Result<_> {
                    let Model { inputs, layers, .. } = model;

                    let input_names: HashSet<_> = inputs
                        .iter()
                        .map(|(input_name, _shape)| input_name)
                        .collect();
                    let layer_names: HashSet<_> = layers
                        .iter()
                        .filter_map(|(layer_name, _layer)| layer_name.name())
                        .collect();

                    layers
                        .iter()
                        .flat_map(|(self_name, layer)| {
                            layer
                                .input_paths()
                                .into_iter()
                                .map(move |input_path| (self_name, input_path))
                        })
                        .try_for_each(|(self_name, input_path)| -> Result<_> {
                            match input_path {
                                ModulePath::Layer(input_name) => {
                                    // refer to input or any other layer
                                    let self_reference = self_name
                                        .name()
                                        .map(|self_name| self_name == input_name)
                                        .unwrap_or(false);
                                    ensure!(
                                        !self_reference,
                                        "self-referencing to '{}' is not allowed",
                                        input_name
                                    );

                                    let is_valid = input_names.contains(&input_name)
                                        || layer_names.contains(&input_name);
                                    ensure!(
                                        is_valid,
                                        "'{}' does not refer to an input or a layer",
                                        input_name
                                    );
                                }
                                ModulePath::GroupRef {
                                    layer: input_name,
                                    output: output_name,
                                } => {
                                    // refer to a "Group" layer
                                    let self_reference = self_name
                                        .name()
                                        .map(|self_name| self_name == input_name)
                                        .unwrap_or(false);
                                    ensure!(
                                        !self_reference,
                                        "self-referencing to '{}' is not allowed",
                                        input_name
                                    );

                                    // check referred group has the specified output name
                                    let input_layer = layers
                                        .get(&ModuleIdent::from(input_name.clone()))
                                        .ok_or_else(|| {
                                            format_err!(
                                                "'{}' does not refer to a layer",
                                                input_name
                                            )
                                        })?;

                                    match input_layer {
                                        Module::Group(Group {
                                            group: group_name, ..
                                        }) => {
                                            debug_assert!(self_group_name != group_name);

                                            match (
                                                parent_groups.0.get(group_name),
                                                local_groups.get(group_name),
                                            ) {
                                                (Some(group), None) | (None, Some(group)) => {
                                                    let has_output =
                                                        group.outputs.contains_key(output_name);
                                                    ensure!(
                                                    has_output,
                                                    "the group '{}' does not have the output '{}'",
                                                    group_name,
                                                    output_name
                                                );
                                                }
                                                _ => unreachable!(),
                                            }
                                        }
                                        _ => bail!(
                                            "the referred layer by '{}' is not a Group module",
                                            input_name
                                        ),
                                    }
                                }
                            }

                            Ok(())
                        })?;

                    Ok(())
                })?;

            // merge with included groups
            let merged_groups = Self::merge(vec![parent_groups, ModelGroups(local_groups)])?;

            Ok(merged_groups)
        }

        pub fn merge(groups_iter: impl IntoIterator<Item = Self>) -> Result<Self> {
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
    }
}

mod model {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    #[serde(try_from = "ModelUnchecked", into = "ModelUnchecked")]
    pub struct Model {
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<ModuleName, Shape, _>"))]
        pub inputs: IndexMap<ModuleName, Shape>,
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<ModuleIdent, Module, _>"))]
        pub layers: IndexMap<ModuleIdent, Module>,
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<ModuleName, ModulePath, _>"))]
        pub outputs: IndexMap<ModuleName, ModulePath>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    struct ModelUnchecked {
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<ModuleName, Shape, _>"))]
        pub inputs: IndexMap<ModuleName, Shape>,
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<ModuleIdent, Module, _>"))]
        pub layers: IndexMap<ModuleIdent, Module>,
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<ModuleName, ModulePath, _>"))]
        pub outputs: IndexMap<ModuleName, ModulePath>,
    }

    impl TryFrom<ModelUnchecked> for Model {
        type Error = Error;

        fn try_from(from: ModelUnchecked) -> Result<Self, Self::Error> {
            let ModelUnchecked {
                inputs,
                layers,
                outputs,
            } = from;

            // check duplicated layer names
            layers
                .keys()
                .filter_map(|ident| ident.name())
                .map(|name| name.as_ref())
                .chain(inputs.keys().map(AsRef::as_ref))
                .chain(outputs.keys().map(AsRef::as_ref))
                .try_fold(HashSet::new(), |mut set, name| {
                    ensure!(set.insert(name), "duplicated layer name '{}'", name);
                    Ok(set)
                })?;

            Ok(Self {
                inputs,
                layers,
                outputs,
            })
        }
    }

    impl From<Model> for ModelUnchecked {
        fn from(from: Model) -> Self {
            let Model {
                inputs,
                layers,
                outputs,
            } = from;

            Self {
                inputs,
                layers,
                outputs,
            }
        }
    }
}
