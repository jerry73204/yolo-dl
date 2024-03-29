use super::LayerIndex;
use crate::common::*;
use num_traits::{NumCast, ToPrimitive};

// pub fn transpose_matrix<T>(buf: &mut [T], nrows: usize, ncols: usize) -> Result<()>
// where
//     T: Clone,
// {
//     ensure!(buf.len() == nrows * ncols, "the size does not match");
//     let tmp = buf.to_owned();

//     (0..nrows).for_each(|row| {
//         (0..ncols).for_each(|col| {
//             buf[col * nrows + row] = tmp[row * ncols + col].clone();
//         });
//     });

//     Ok(())
// }

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct DisplayAsDebug<T>(pub T)
where
    T: Display;

impl<T> Debug for DisplayAsDebug<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

pub use from_layers::*;
mod from_layers {
    use super::*;

    #[derive(Debug, Clone, Default)]
    pub struct FromLayers(pub IndexSet<LayerIndex>);

    impl Deref for FromLayers {
        type Target = IndexSet<LayerIndex>;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl PartialEq for FromLayers {
        fn eq(&self, other: &Self) -> bool {
            self.len() == other.len() && izip!(&self.0, &other.0).all(|(lhs, rhs)| lhs == rhs)
        }
    }

    impl Eq for FromLayers {}

    impl Hash for FromLayers {
        fn hash<H>(&self, state: &mut H)
        where
            H: Hasher,
        {
            self.iter().for_each(|item| {
                item.hash(state);
            });
        }
    }

    impl Serialize for FromLayers {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let text = self
                .0
                .iter()
                .cloned()
                .map(|index| index.ordinal().to_string())
                .join(",");
            text.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for FromLayers {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let text = String::deserialize(deserializer)?;
            let layers_vec: Vec<_> = text
                .split(',')
                .map(|token| -> Result<_, String> {
                    let index: isize = token
                        .trim()
                        .parse()
                        .map_err(|_| format!("{} is not a valid index", token))?;
                    let index = LayerIndex::from_ordinal(index);
                    Ok(index)
                })
                .try_collect()
                .map_err(|err| {
                    D::Error::custom(format!("failed to parse layer index: {:?}", err))
                })?;
            let layers_set: IndexSet<LayerIndex> = layers_vec.iter().cloned().collect();

            if layers_vec.len() != layers_set.len() {
                return Err(D::Error::custom("duplicated layer index is not allowed"));
            }

            Ok(Self(layers_set))
        }
    }
}

pub fn default<T>() -> T
where
    T: Default,
{
    T::default()
}

pub fn bool_true() -> bool {
    true
}

pub fn bool_false() -> bool {
    false
}

pub mod zero_one_bool {
    use super::*;

    pub fn serialize<S>(&yes: &bool, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (yes as isize).serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<bool, D::Error>
    where
        D: Deserializer<'de>,
    {
        match isize::deserialize(deserializer)? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(D::Error::invalid_value(
                serde::de::Unexpected::Signed(value as i64),
                &"0 or 1",
            )),
        }
    }
}

pub fn integer<T, const VALUE: usize>() -> T
where
    T: NumCast,
{
    <T as NumCast>::from(VALUE).unwrap()
}

pub fn ratio<T, const NUM: usize, const DENO: usize>() -> T
where
    T: NumCast,
{
    let ratio = NUM.to_f64().unwrap() / DENO.to_f64().unwrap();
    <T as NumCast>::from(ratio).unwrap()
}

pub mod serde_r64_comma_list {
    use super::*;

    pub fn serialize<S>(values: &Option<Vec<R64>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        values
            .as_ref()
            .map(|values| {
                let text: String = values
                    .iter()
                    .cloned()
                    .map(|val| val.raw().to_string())
                    .join(",");
                text
            })
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Vec<R64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = Option::<String>::deserialize(deserializer)?;
        let values: Option<Vec<_>> = text
            .map(|text| {
                let values: Result<Vec<_>, _> = text
                    .split(',')
                    .map(|token| {
                        let value: f64 = token.trim().parse().map_err(|_| {
                            D::Error::custom(format!("unable to parse token '{}'", token))
                        })?;
                        let value = R64::try_new(value).ok_or_else(|| {
                            D::Error::custom(format!("invalid value '{}'", token))
                        })?;
                        Ok(value)
                    })
                    .collect();
                values
            })
            .transpose()?;
        Ok(values)
    }
}

pub mod serde_comma_list {
    use super::*;

    pub fn serialize<S, T>(values: &Option<Vec<T>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        T: ToString,
        S: Serializer,
    {
        values
            .as_ref()
            .map(|values| values.iter().map(|val| val.to_string()).join(","))
            .serialize(serializer)
    }

    pub fn deserialize<'de, D, T>(deserializer: D) -> Result<Option<Vec<T>>, D::Error>
    where
        T: FromStr,
        D: Deserializer<'de>,
    {
        let text = Option::<String>::deserialize(deserializer)?;
        let values: Option<Vec<_>> = text
            .map(|text| {
                let values: Result<Vec<_>, _> = text
                    .split(',')
                    .map(|token| {
                        token.trim().parse().map_err(|_| {
                            D::Error::custom(format!("unable to parse token '{}'", token))
                        })
                    })
                    .collect();
                values
            })
            .transpose()?;
        Ok(values)
    }
}

pub mod serde_anchors {
    use super::*;

    pub fn serialize<S>(
        steps: &Option<Vec<(usize, usize)>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        steps
            .as_ref()
            .map(|steps| {
                steps
                    .iter()
                    .flat_map(|(w, h)| [w, h])
                    .map(|val| val.to_string())
                    .join(",")
            })
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Vec<(usize, usize)>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let values: Option<Vec<usize>> = serde_comma_list::deserialize(deserializer)?;
        let values = match values {
            Some(values) => values,
            None => return Ok(None),
        };

        if values.len() % 2 != 0 {
            return Err(D::Error::custom("expect even number of values"));
        }

        let anchors: Vec<_> = values
            .into_iter()
            .chunks(2)
            .into_iter()
            .map(|mut chunk| (chunk.next().unwrap(), chunk.next().unwrap()))
            .collect();
        Ok(Some(anchors))
    }
}
