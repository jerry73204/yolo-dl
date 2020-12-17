use super::{
    config::ModelGroupsConfig,
    module::{
        Bottleneck, BottleneckCsp, Concat, ConvBlock, Detect, Focus, Group, Input, Output, Spp,
        UpSample,
    },
};
use crate::{common::*, utils};

pub use group_name::*;
pub use layer::*;
pub use layer_ident::*;
pub use layer_name::*;
pub use layer_path::*;
pub use model::*;

mod layer_name {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct LayerName(String);

    impl LayerName {
        pub fn new<'a>(name: impl Into<Cow<'a, str>>) -> Result<Self> {
            let name = name.into().into_owned();
            ensure!(!name.is_empty(), "layer name must not be empty");
            ensure!(!name.contains('.'), "layer name must not contain dot '.'");
            Ok(Self(name))
        }
    }

    impl Serialize for LayerName {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            self.0.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for LayerName {
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

    impl TryFrom<&str> for LayerName {
        type Error = Error;

        fn try_from(name: &str) -> Result<Self, Self::Error> {
            Self::new(name)
        }
    }

    impl Borrow<str> for LayerName {
        fn borrow(&self) -> &str {
            self.0.as_ref()
        }
    }

    impl AsRef<str> for LayerName {
        fn as_ref(&self) -> &str {
            self.0.as_ref()
        }
    }

    impl Display for LayerName {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            self.0.fmt(f)
        }
    }
}

mod group_name {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct GroupName(String);

    impl GroupName {
        pub fn new<'a>(name: impl Into<Cow<'a, str>>) -> Result<Self> {
            let name = name.into().into_owned();
            ensure!(!name.is_empty(), "layer name must not be empty");
            ensure!(!name.contains('.'), "layer name must not contain dot '.'");
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

mod layer_ident {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub enum LayerIdent {
        Unnamed,
        Named(LayerName),
    }

    impl LayerIdent {
        pub fn new<'a>(name: impl Into<Cow<'a, str>>) -> Result<Self> {
            let name = name.into();
            let name: &str = name.borrow();
            let ident = match name {
                "" => Self::Unnamed,
                _ => Self::Named(name.try_into()?),
            };
            Ok(ident)
        }

        pub fn name(&self) -> Option<&str> {
            match self {
                Self::Unnamed => None,
                Self::Named(name) => Some(name.as_ref()),
            }
        }
    }

    impl Serialize for LayerIdent {
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

    impl<'de> Deserialize<'de> for LayerIdent {
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

    impl TryFrom<&str> for LayerIdent {
        type Error = Error;

        fn try_from(name: &str) -> Result<Self, Self::Error> {
            Self::new(name)
        }
    }

    impl Display for LayerIdent {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            match self {
                Self::Unnamed => "[unnamed]",
                Self::Named(name) => name.as_ref(),
            }
            .fmt(f)
        }
    }
}

mod layer_path {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub enum LayerPath {
        Layer(LayerName),
        GroupRef { layer: LayerName, output: LayerName },
    }

    impl LayerPath {
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

    impl Serialize for LayerPath {
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

    impl<'de> Deserialize<'de> for LayerPath {
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

    impl TryFrom<&str> for LayerPath {
        type Error = Error;

        fn try_from(name: &str) -> Result<Self, Self::Error> {
            Self::new(name)
        }
    }

    impl Display for LayerPath {
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

    fn load_recursive(path: impl AsRef<Path>, max_depth: usize, curr_depth: usize) -> Result<Self> {
        debug_assert!(max_depth > 0, "max_depth must be positive");
        ensure!(curr_depth < max_depth, "max_depth exceeded");

        let ModelGroupsConfig { includes, groups } = ModelGroupsConfig::from_path(path)?;

        // load included groups
        let included_groups_vec: Vec<_> = includes
            .into_iter()
            .map(|path| Self::load_recursive(path, max_depth, curr_depth + 1))
            .try_collect()?;
        let parent_groups = Self::merge(included_groups_vec)?;

        // check if group layer refer to valid groups
        groups
            .iter()
            .flat_map(|(_group_name, model)| {
                let Model { layers, .. } = model;
                layers.iter()
            })
            .filter_map(|(_layer_name, layer)| match layer {
                Layer::Group(Group { group, .. }) => Some(group),
                _ => None,
            })
            .try_for_each(|group_name| -> Result<_> {
                let Self(parent_groups) = &parent_groups;
                ensure!(
                    parent_groups.contains_key(group_name),
                    "'{}' is not a valid group name",
                    group_name
                );
                Ok(())
            })?;

        // merge with included groups
        let merged_groups = Self::merge(vec![ModelGroups(groups), parent_groups])?;

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

mod model {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    #[serde(try_from = "ModelUnchecked", into = "ModelUnchecked")]
    pub struct Model {
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<LayerName, Input, _>"))]
        pub inputs: IndexMap<LayerName, Input>,
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<LayerIdent, Layer, _>"))]
        pub layers: IndexMap<LayerIdent, Layer>,
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<LayerName, Output, _>"))]
        pub outputs: IndexMap<LayerName, Output>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    struct ModelUnchecked {
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<LayerName, Input, _>"))]
        pub inputs: IndexMap<LayerName, Input>,
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<LayerIdent, Layer, _>"))]
        pub layers: IndexMap<LayerIdent, Layer>,
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<LayerName, Output, _>"))]
        pub outputs: IndexMap<LayerName, Output>,
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

mod layer {
    use super::*;

    pub trait LayerEx {
        fn input_layers(&self) -> Vec<&LayerPath>;
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(tag = "kind")]
    pub enum Layer {
        Focus(Focus),
        ConvBlock(ConvBlock),
        Bottleneck(Bottleneck),
        BottleneckCsp(BottleneckCsp),
        Spp(Spp),
        UpSample(UpSample),
        Concat(Concat),
        Detect(Detect),
        Group(Group),
    }

    impl Layer {
        pub fn output_shape(&self, _input_shape: &[usize]) -> Option<Vec<usize>> {
            // match &self.kind {
            //     LayerKind::Input(layer) => layer.output_shape(input_shape),
            //     LayerKind::Focus(layer) => layer.output_shape(input_shape),
            //     LayerKind::ConvBlock(layer) => layer.output_shape(input_shape),
            //     LayerKind::Bottleneck(layer) => layer.output_shape(input_shape),
            //     LayerKind::BottleneckCsp(layer) => layer.output_shape(input_shape),
            //     LayerKind::Spp(layer) => layer.output_shape(input_shape),
            //     LayerKind::UpSample(layer) => layer.output_shape(input_shape),
            //     LayerKind::Concat(layer) => layer.output_shape(input_shape),
            //     LayerKind::Detect(layer) => layer.output_shape(input_shape),
            // }
            todo!();
        }
    }

    impl LayerEx for Layer {
        fn input_layers(&self) -> Vec<&LayerPath> {
            match self {
                Layer::Focus(layer) => layer.input_layers(),
                Layer::ConvBlock(layer) => layer.input_layers(),
                Layer::Bottleneck(layer) => layer.input_layers(),
                Layer::BottleneckCsp(layer) => layer.input_layers(),
                Layer::Spp(layer) => layer.input_layers(),
                Layer::UpSample(layer) => layer.input_layers(),
                Layer::Concat(layer) => layer.input_layers(),
                Layer::Detect(layer) => layer.input_layers(),
                Layer::Group(layer) => layer.input_layers(),
            }
        }
    }
}
