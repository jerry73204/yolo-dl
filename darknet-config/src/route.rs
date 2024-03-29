use super::Meta;
use crate::{common::*, utils, utils::FromLayers};

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
            group_id: group.group_id(),
            groups: NonZeroUsize::new(group.num_groups()).unwrap(),
            common,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RouteGroup {
    group_id: usize,
    num_groups: usize,
}

impl RouteGroup {
    pub fn new(group_id: usize, num_groups: usize) -> Option<Self> {
        if num_groups == 0 || group_id >= num_groups {
            None
        } else {
            Some(Self {
                group_id,
                num_groups,
            })
        }
    }

    pub fn group_id(&self) -> usize {
        self.group_id
    }

    pub fn num_groups(&self) -> usize {
        self.num_groups
    }
}

fn default_groups() -> NonZeroUsize {
    NonZeroUsize::new(1).unwrap()
}
