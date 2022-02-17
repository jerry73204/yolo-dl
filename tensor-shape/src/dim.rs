use crate::common::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dim {
    Size(usize),
    Infer,
}

impl Dim {
    pub fn size(&self) -> Option<usize> {
        Option::<usize>::from(*self)
    }

    pub fn is_compatible_with(&self, other: &Dim) -> bool {
        match (self, other) {
            (Self::Size(lhs), Self::Size(rhs)) => lhs == rhs,
            _ => true,
        }
    }

    pub fn equalize(&self, other: &Dim) -> Option<Self> {
        match (self, other) {
            (Self::Size(lhs), Self::Size(rhs)) => {
                if lhs == rhs {
                    Some(*self)
                } else {
                    None
                }
            }
            (Self::Size(_), Self::Infer) => Some(*self),
            (Self::Infer, Self::Size(_)) => Some(*other),
            (Self::Infer, Self::Infer) => Some(*self),
        }
    }

    pub fn scale(&self, scale: R64) -> Self {
        match *self {
            Self::Size(size) => {
                let new_size = (scale * size as f64).floor().raw() as usize;
                Self::Size(new_size)
            }
            Self::Infer => Self::Infer,
        }
    }
}

impl From<usize> for Dim {
    fn from(from: usize) -> Self {
        Self::Size(from)
    }
}

impl From<Option<usize>> for Dim {
    fn from(from: Option<usize>) -> Self {
        match from {
            Some(size) => Self::Size(size),
            None => Self::Infer,
        }
    }
}

impl From<Dim> for Option<usize> {
    fn from(from: Dim) -> Self {
        match from {
            Dim::Size(size) => Some(size),
            Dim::Infer => None,
        }
    }
}

impl Serialize for Dim {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Size(value) => value.serialize(serializer),
            Self::Infer => "_".serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for Dim {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde_json::Value;

        let value = Value::deserialize(deserializer)?;
        let dim = match value {
            Value::String(text) => {
                if text != "_" {
                    return Err(D::Error::custom(format!("'{}' is not a dimension", text)));
                }
                Self::Infer
            }
            Value::Number(value) => {
                let value = value
                    .as_u64()
                    .ok_or_else(|| D::Error::custom(format!("'{}' is not a dimension", value)))?;
                Self::Size(value as usize)
            }
            value => {
                return Err(D::Error::custom(format!("'{}' is not a dimension", value)));
            }
        };
        Ok(dim)
    }
}

impl Add<Dim> for Dim {
    type Output = Dim;

    fn add(self, rhs: Dim) -> Self::Output {
        match (self, rhs) {
            (Self::Size(lhs), Self::Size(rhs)) => Self::Size(lhs + rhs),
            _ => Self::Infer,
        }
    }
}

impl Sub<Dim> for Dim {
    type Output = Dim;

    fn sub(self, rhs: Dim) -> Self::Output {
        match (self, rhs) {
            (Self::Size(lhs), Self::Size(rhs)) => Self::Size(lhs - rhs),
            _ => Self::Infer,
        }
    }
}

impl Mul<Dim> for Dim {
    type Output = Dim;

    fn mul(self, rhs: Dim) -> Self::Output {
        match (self, rhs) {
            (Self::Size(lhs), Self::Size(rhs)) => Self::Size(lhs * rhs),
            _ => Self::Infer,
        }
    }
}

impl Div<Dim> for Dim {
    type Output = Dim;

    fn div(self, rhs: Dim) -> Self::Output {
        match (self, rhs) {
            (Self::Size(lhs), Self::Size(rhs)) => Self::Size(lhs / rhs),
            _ => Self::Infer,
        }
    }
}

impl Add<usize> for Dim {
    type Output = Dim;

    fn add(self, rhs: usize) -> Self::Output {
        match self {
            Self::Size(lhs) => Self::Size(lhs + rhs),
            _ => Self::Infer,
        }
    }
}

impl Sub<usize> for Dim {
    type Output = Dim;

    fn sub(self, rhs: usize) -> Self::Output {
        match self {
            Self::Size(lhs) => Self::Size(lhs - rhs),
            _ => Self::Infer,
        }
    }
}

impl Mul<usize> for Dim {
    type Output = Dim;

    fn mul(self, rhs: usize) -> Self::Output {
        match self {
            Self::Size(lhs) => Self::Size(lhs * rhs),
            _ => Self::Infer,
        }
    }
}

impl Div<usize> for Dim {
    type Output = Dim;

    fn div(self, rhs: usize) -> Self::Output {
        match self {
            Self::Size(lhs) => Self::Size(lhs / rhs),
            _ => Self::Infer,
        }
    }
}

impl Add<Dim> for usize {
    type Output = Dim;

    fn add(self, rhs: Dim) -> Self::Output {
        match rhs {
            Dim::Size(rhs) => Dim::Size(self + rhs),
            _ => Dim::Infer,
        }
    }
}

impl Sub<Dim> for usize {
    type Output = Dim;

    fn sub(self, rhs: Dim) -> Self::Output {
        match rhs {
            Dim::Size(rhs) => Dim::Size(self - rhs),
            _ => Dim::Infer,
        }
    }
}

impl Mul<Dim> for usize {
    type Output = Dim;

    fn mul(self, rhs: Dim) -> Self::Output {
        match rhs {
            Dim::Size(rhs) => Dim::Size(self * rhs),
            _ => Dim::Infer,
        }
    }
}

impl Div<Dim> for usize {
    type Output = Dim;

    fn div(self, rhs: Dim) -> Self::Output {
        match rhs {
            Dim::Size(rhs) => Dim::Size(self / rhs),
            _ => Dim::Infer,
        }
    }
}
