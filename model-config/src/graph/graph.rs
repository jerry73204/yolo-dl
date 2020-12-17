use crate::{common::*, utils};

pub use graph::*;
pub use node::*;

mod graph {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    pub struct Graph {
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<usize, Node, _>"))]
        pub nodes: IndexMap<usize, Node>,
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<usize, Edge, _>"))]
        pub edges: IndexMap<usize, Edge>,
    }

    // impl Graph {
    //     pub fn from_config(config: &Config) -> Self {
    //         todo!();
    //     }
    // }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum Node {
        Input(Input),
        Output(Output),
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Edge {
        pub from: usize,
        pub to: usize,
        pub shape: Vec<usize>,
    }
}

mod node {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Input {
        pub shape: Vec<usize>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Output {}

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Focus {}

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct ConvBlock {}

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Bottleneck {}

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct BottleneckCsp {}

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Spp {}

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct UpSample {}

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Concat {}

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Detect {}
}
