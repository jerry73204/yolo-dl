use crate::{common::*, utils};
use model_config::{Module, ModulePath, ShapeOutput};

#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(PartialEq, Eq, Hash)]
pub struct Graph {
    #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<NodeKey, Node, _>"))]
    pub(crate) nodes: IndexMap<NodeKey, Node>,
}

impl Graph {
    /// Get a reference to the graph's nodes.
    pub fn nodes(&self) -> &IndexMap<NodeKey, Node> {
        &self.nodes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Node {
    pub input_keys: InputKeys,
    pub output_shape: ShapeOutput,
    pub path: Option<ModulePath>,
    pub config: Module,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputKeys {
    None,
    PlaceHolder,
    Single(NodeKey),
    Indexed(Vec<NodeKey>),
}

impl InputKeys {
    pub fn iter(&self) -> impl Iterator<Item = NodeKey> {
        let iter: Box<dyn Iterator<Item = NodeKey>> = match *self {
            Self::None => Box::new(iter::empty()),
            Self::PlaceHolder => Box::new(iter::empty()),
            Self::Single(key) => Box::new(iter::once(key)),
            Self::Indexed(ref keys) => Box::new(keys.clone().into_iter()),
        };
        iter
    }

    pub fn single(&self) -> Option<NodeKey> {
        match *self {
            Self::Single(key) => Some(key),
            _ => None,
        }
    }

    pub fn indexed(&self) -> Option<&[NodeKey]> {
        match self {
            Self::Indexed(keys) => Some(keys.as_slice()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeKey(pub usize);

impl Display for NodeKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}
