use super::{LayerIndex, Meta, RouteGroup};
use crate::{common::*, utils, utils::FromLayers};

#[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
#[derivative(Hash)]
#[serde(try_from = "RawRoute", into = "RawRoute")]
pub struct Route {
    pub layers: FromLayers,
    pub group: RouteGroup,
    pub common: Meta,
}

impl Route {
    pub fn output_shape(&self, input_shapes: &[[usize; 3]]) -> Option<[usize; 3]> {
        let [out_h, out_w] = {
            let set: HashSet<_> = input_shapes.iter().map(|&[h, w, _c]| [h, w]).collect();
            let mut iter = set.into_iter();
            let first = iter.next()?;
            if iter.next().is_some() {
                return None;
            }
            first
        };

        let num_groups = self.group.num_groups();

        let out_c: usize = input_shapes.iter().try_fold(0, |sum, &[_h, _w, in_c]| {
            (in_c % num_groups == 0).then(|| sum + in_c / num_groups)
        })?;

        Some([out_h, out_w, out_c])
    }
}

impl TryFrom<RawRoute> for Route {
    type Error = Error;

    fn try_from(from: RawRoute) -> Result<Self, Self::Error> {
        let RawRoute {
            layers,
            group_id,
            groups,
            common,
        } = from;

        let group = RouteGroup::new(group_id, groups.get())
            .ok_or_else(|| anyhow!("group_id must be less than groups"))?;

        Ok(Self {
            layers,
            group,
            common,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
#[derivative(Hash)]
pub(super) struct RawRoute {
    pub layers: FromLayers,
    #[serde(default = "default_groups")]
    pub groups: NonZeroUsize,
    #[serde(default = "utils::integer::<_, 0>")]
    pub group_id: usize,
    #[serde(flatten)]
    pub common: Meta,
}

impl From<Route> for RawRoute {
    fn from(from: Route) -> Self {
        let Route {
            layers,
            group,
            common,
        } = from;

        Self {
            layers,
            group_id: group.group_id().into(),
            groups: NonZeroUsize::new(group.num_groups()).unwrap().into(),
            common,
        }
    }
}

pub mod serde_layers {
    use super::*;

    pub fn serialize<S>(indexes: &IndexSet<LayerIndex>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let text = indexes
            .iter()
            .cloned()
            .map(|index| isize::from(index).to_string())
            .join(",");
        text.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<IndexSet<LayerIndex>, D::Error>
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
            .map_err(|err| D::Error::custom(format!("failed to parse layer index: {:?}", err)))?;
        let layers_set: IndexSet<LayerIndex> = layers_vec.iter().cloned().collect();

        if layers_vec.len() != layers_set.len() {
            return Err(D::Error::custom("duplicated layer index is not allowed"));
        }

        Ok(layers_set)
    }
}

fn default_groups() -> NonZeroUsize {
    NonZeroUsize::new(1).unwrap()
}
