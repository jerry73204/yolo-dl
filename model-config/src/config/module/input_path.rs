use super::*;
use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GroupPath {
    pub layer: ModuleName,
    pub output: ModuleName,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModulePath(Vec<ModuleName>);

impl ModulePath {
    pub fn empty() -> Self {
        Self(vec![])
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn depth(&self) -> usize {
        self.0.len()
    }

    pub fn extend(&self, other: &ModulePath) -> Self {
        self.0.iter().chain(other.0.iter()).collect()
    }

    pub fn join<'a>(&self, name: impl Into<Cow<'a, ModuleName>>) -> Self {
        let name = name.into().into_owned();
        self.0.iter().cloned().chain(iter::once(name)).collect()
    }
}

impl FromStr for ModulePath {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        let tokens = name.split('.');
        let components: Vec<_> = tokens
            .map(|token| ModuleName::from_str(token))
            .try_collect()?;
        Ok(Self(components))
    }
}

impl Serialize for ModulePath {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_string().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ModulePath {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        let ident = Self::from_str(&text)
            .map_err(|err| D::Error::custom(format!("invalid name: {:?}", err)))?;
        Ok(ident)
    }
}

impl Display for ModulePath {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let text = self.0.iter().map(AsRef::as_ref).join(".");
        Display::fmt(&text, f)
    }
}

impl AsRef<[ModuleName]> for ModulePath {
    fn as_ref(&self) -> &[ModuleName] {
        self.0.as_ref()
    }
}

impl FromIterator<ModuleName> for ModulePath {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = ModuleName>,
    {
        Self(Vec::from_iter(iter))
    }
}

impl<'a> FromIterator<&'a ModuleName> for ModulePath {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'a ModuleName>,
    {
        Self(iter.into_iter().cloned().collect())
    }
}
