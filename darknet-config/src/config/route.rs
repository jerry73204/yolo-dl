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
    pub groups: NonZeroU64,
    #[serde(default = "defaults::route_group_id")]
    pub group_id: u64,
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
            groups: NonZeroU64::new(group.num_groups()).unwrap(),
            common,
        }
    }
}
