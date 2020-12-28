use crate::common::*;

pub use dim::*;
pub use shape::*;
pub use size::*;

mod dim {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Dim(Option<usize>);

    impl Dim {
        pub fn size(&self) -> Option<usize> {
            self.0
        }

        pub fn is_compatible_with(&self, other: &Dim) -> bool {
            match (self.0, other.0) {
                (Some(lhs), Some(rhs)) => lhs == rhs,
                _ => true,
            }
        }

        pub fn equalize(&self, other: &Dim) -> Option<Self> {
            match (self.0, other.0) {
                (Some(lhs), Some(rhs)) => {
                    if lhs == rhs {
                        Some(*self)
                    } else {
                        None
                    }
                }
                (Some(_), None) => Some(*self),
                (None, Some(_)) => Some(*other),
                (None, None) => Some(*self),
            }
        }
    }

    impl From<usize> for Dim {
        fn from(from: usize) -> Self {
            Self(Some(from))
        }
    }

    impl From<Option<usize>> for Dim {
        fn from(from: Option<usize>) -> Self {
            Self(from)
        }
    }

    impl From<Dim> for Option<usize> {
        fn from(from: Dim) -> Self {
            from.0
        }
    }

    impl Serialize for Dim {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            match self.0 {
                Some(value) => value.serialize(serializer),
                None => "_".serialize(serializer),
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
                    Self(None)
                }
                Value::Number(value) => {
                    let value = value.as_u64().ok_or_else(|| {
                        D::Error::custom(format!("'{}' is not a dimension", value))
                    })?;
                    Self(Some(value as usize))
                }
                value => {
                    return Err(D::Error::custom(format!("'{}' is not a dimension", value)));
                }
            };
            Ok(dim)
        }
    }
}

mod shape {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct Shape(Vec<Dim>);

    impl Shape {
        pub fn is_compatible_with(&self, other: &Shape) -> bool {
            if self.0.len() != other.0.len() {
                return false;
            }

            self.0
                .iter()
                .zip(other.0.iter())
                .all(|(lhs, rhs)| lhs.is_compatible_with(rhs))
        }

        pub fn equalize(&self, other: &Shape) -> Option<Shape> {
            if self.0.len() != other.0.len() {
                return None;
            }

            let new_shape: Option<Vec<Dim>> = self
                .0
                .iter()
                .zip(other.0.iter())
                .map(|(lhs, rhs)| lhs.equalize(rhs))
                .collect();
            let new_shape = new_shape?;
            Some(new_shape.into())
        }
    }

    impl From<Vec<Option<usize>>> for Shape {
        fn from(vec: Vec<Option<usize>>) -> Self {
            Self(vec.into_iter().map(Dim::from).collect())
        }
    }

    impl From<&[Option<usize>]> for Shape {
        fn from(slice: &[Option<usize>]) -> Self {
            Vec::from(slice).into()
        }
    }

    impl From<Vec<usize>> for Shape {
        fn from(vec: Vec<usize>) -> Self {
            Self(vec.into_iter().map(Dim::from).collect())
        }
    }

    impl From<Vec<Dim>> for Shape {
        fn from(vec: Vec<Dim>) -> Self {
            Self(vec)
        }
    }

    impl From<&[usize]> for Shape {
        fn from(slice: &[usize]) -> Self {
            Vec::from(slice).into()
        }
    }

    impl From<&Shape> for Vec<Option<usize>> {
        fn from(shape: &Shape) -> Self {
            shape.0.iter().cloned().map(Into::into).collect()
        }
    }

    impl AsRef<[Dim]> for Shape {
        fn as_ref(&self) -> &[Dim] {
            &self.0
        }
    }

    impl Display for Shape {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            let mut list = f.debug_list();
            self.0.iter().for_each(|dim| match dim.size() {
                Some(value) => {
                    list.entry(&value);
                }
                None => {
                    list.entry(&'_');
                }
            });
            list.finish()
        }
    }
}

mod size {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(from = "(R64, R64)", into = "(R64, R64)")]
    pub struct Size {
        pub h: R64,
        pub w: R64,
    }

    impl From<(R64, R64)> for Size {
        fn from((h, w): (R64, R64)) -> Self {
            Self { h, w }
        }
    }

    impl From<Size> for (R64, R64) {
        fn from(Size { h, w }: Size) -> Self {
            (h, w)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Activation {
    #[serde(rename = "mish")]
    Mish,
    #[serde(rename = "hard_mish")]
    HardMish,
    #[serde(rename = "swish")]
    Swish,
    #[serde(rename = "normalize_channels")]
    NormalizeChannels,
    #[serde(rename = "normalize_channels_softmax")]
    NormalizeChannelsSoftmax,
    #[serde(rename = "normalize_channels_softmax_maxval")]
    NormalizeChannelsSoftmaxMaxval,
    #[serde(rename = "logistic")]
    Logistic,
    #[serde(rename = "loggy")]
    Loggy,
    #[serde(rename = "relu")]
    Relu,
    #[serde(rename = "elu")]
    Elu,
    #[serde(rename = "selu")]
    Selu,
    #[serde(rename = "gelu")]
    Gelu,
    #[serde(rename = "relie")]
    Relie,
    #[serde(rename = "ramp")]
    Ramp,
    #[serde(rename = "linear")]
    Linear,
    #[serde(rename = "tanh")]
    Tanh,
    #[serde(rename = "plse")]
    Plse,
    #[serde(rename = "leaky")]
    Leaky,
    #[serde(rename = "stair")]
    Stair,
    #[serde(rename = "hardtan")]
    Hardtan,
    #[serde(rename = "lhtan")]
    Lhtan,
    #[serde(rename = "relu6")]
    Relu6,
}
