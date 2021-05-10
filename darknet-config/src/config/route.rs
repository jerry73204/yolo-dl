use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
#[derivative(Hash)]
#[serde(try_from = "RawRoute", into = "RawRoute")]
pub struct Route {
    #[derivative(Hash(hash_with = "hash_vec::<LayerIndex, _>"))]
    pub layers: IndexSet<LayerIndex>,
    pub group: RouteGroup,
    pub common: Common,
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
            .ok_or_else(|| format_err!("group_id must be less than groups"))?;

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
    #[derivative(Hash(hash_with = "hash_vec::<LayerIndex, _>"))]
    #[serde(with = "serde_::vec_layers")]
    pub layers: IndexSet<LayerIndex>,
    #[serde(default = "defaults::route_groups")]
    pub groups: NonZeroUsize,
    #[serde(default = "defaults::route_group_id")]
    pub group_id: usize,
    #[serde(flatten)]
    pub common: Common,
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
