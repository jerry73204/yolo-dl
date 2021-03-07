use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModuleName(String);

impl FromStr for ModuleName {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self> {
        ensure!(!name.is_empty(), "module name must not be empty");
        ensure!(!name.contains('.'), "module name must not contain dot '.'");
        Ok(Self(name.to_owned()))
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
        let name = Self::from_str(&text)
            .map_err(|err| D::Error::custom(format!("invalid name: {:?}", err)))?;
        Ok(name)
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
        Display::fmt(&self.0, f)
    }
}

impl From<ModuleName> for Cow<'_, ModuleName> {
    fn from(from: ModuleName) -> Self {
        Cow::Owned(from)
    }
}

impl<'a> From<&'a ModuleName> for Cow<'a, ModuleName> {
    fn from(from: &'a ModuleName) -> Self {
        Cow::Borrowed(from)
    }
}
