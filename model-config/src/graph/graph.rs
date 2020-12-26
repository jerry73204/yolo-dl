use crate::{
    common::*,
    config::{
        self, Config, GroupName, LayerEx, LayerIdent, LayerInput, LayerName, LayerPath, Model,
        ModelGroups,
    },
    utils::{self, IteratorEx},
};

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

    impl Graph {
        pub fn load(config: &Config) -> Result<Self> {
            let Config {
                include,
                group: group_name,
            } = config;
            let groups = ModelGroups::load(&include)?;
            let graph = Self::from_model_groups(&groups, group_name)?;
            Ok(graph)
        }

        fn from_model_groups(groups: &ModelGroups, main_group_name: &GroupName) -> Result<Self> {
            #[derive(Debug)]
            enum Layer<'a> {
                Input(&'a config::Input),
                Layer(&'a config::Layer),
                Output(&'a config::Output),
            }

            impl Layer<'_> {
                pub fn input_layers(&self) -> LayerInput<'_> {
                    match self {
                        Self::Input(layer) => layer.input_layers(),
                        Self::Output(layer) => layer.input_layers(),
                        Self::Layer(layer) => layer.input_layers(),
                    }
                }
            }

            #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
            struct NodeKey(pub usize);

            #[derive(Debug, Clone, PartialEq, Eq, Hash)]
            struct NodePath(pub Vec<LayerName>);

            impl NodePath {
                pub fn empty() -> Self {
                    Self(vec![])
                }

                pub fn join<'a>(&self, name: impl Into<Cow<'a, LayerName>>) -> Self {
                    let name = name.into().into_owned();
                    self.0.iter().cloned().chain(iter::once(name)).collect()
                }
            }

            impl FromIterator<LayerName> for NodePath {
                fn from_iter<T>(iter: T) -> Self
                where
                    T: IntoIterator<Item = LayerName>,
                {
                    Self(Vec::from_iter(iter))
                }
            }

            impl<'a> FromIterator<&'a LayerName> for NodePath {
                fn from_iter<T>(iter: T) -> Self
                where
                    T: IntoIterator<Item = &'a LayerName>,
                {
                    Self(iter.into_iter().cloned().collect())
                }
            }

            fn traverse<'a>(
                groups: &'a ModelGroups,
                main_group_name: &GroupName,
                prefix: &NodePath,
                global_nodes: &mut HashMap<NodeKey, Layer<'a>>,
                global_edges: &mut HashSet<(NodeKey, NodeKey)>,
                key_enumerator: &mut impl Iterator<Item = NodeKey>,
                path_key_map: &mut HashMap<NodePath, NodeKey>,
            ) -> Result<()> {
                let Model {
                    inputs,
                    layers,
                    outputs,
                } = groups
                    .get(main_group_name)
                    .ok_or_else(|| format_err!("the group '{}' does not exist", main_group_name))?;

                // let name_layer_map = layers
                //     .iter()
                //     .filter_map(|(path, layer)| Some((path.name()?, Layer::Layer(layer))))
                //     .chain(
                //         inputs
                //             .iter()
                //             .map(|(name, layer)| (name, Layer::Input(layer))),
                //     )
                //     .try_fold(HashMap::new(), |mut set, (name, layer)| -> Result<_> {
                //         let prev = set.insert(name, layer);
                //         ensure!(matches!(prev, None), "duplicated layer name '{}'", name);
                //         Ok(set)
                //     })?;

                // index input nodes
                // inputs.iter().for_each(|(layer_name, layer)| {
                //     let path: NodePath = prefix.join(layer_name);
                //     let key = key_enumerator.next().unwrap();

                //     let prev = path_key_map.insert(path, key);
                //     debug_assert!(matches!(prev, None));

                //     nodes.insert(key, Layer::Input(layer));
                // });

                // toposort layers
                // let sorted_layers = {
                //     #[derive(Debug, Clone, Copy, Derivative)]
                //     #[derivative(PartialOrd, Ord, PartialEq, Eq, Hash)]
                //     struct IndexedLayer<'a> {
                //         index: usize,
                //         #[derivative(
                //             PartialOrd = "ignore",
                //             Ord = "ignore",
                //             PartialEq = "ignore",
                //             Hash = "ignore"
                //         )]
                //         layer_ident: &'a LayerIdent,
                //         #[derivative(
                //             PartialOrd = "ignore",
                //             Ord = "ignore",
                //             PartialEq = "ignore",
                //             Hash = "ignore"
                //         )]
                //         layer: &'a config::Layer,
                //     };

                //     // create nodes
                //     let iter = layers
                //         .iter()
                //         .enumerate()
                //         .map(|(index, (layer_ident, layer))| {
                //             let node = IndexedLayer {
                //                 index,
                //                 layer_ident,
                //                 layer,
                //             };
                //             (layer_ident.name(), node)
                //         });
                //     let nodes: Vec<_> = iter.clone().map(|(_layer_name, node)| node).collect();
                //     let name_node_map: HashMap<_, _> = iter
                //         .filter_map(|(layer_name, node)| Some((layer_name?, node)))
                //         .collect();

                //     // create edges
                //     let edges: Vec<_> = nodes
                //         .iter()
                //         .cloned()
                //         .scan(None, |prev_node, curr_node| {
                //             let saved_prev_node = *prev_node;
                //             *prev_node = Some(curr_node);
                //             Some(Ok((saved_prev_node, curr_node)))
                //         })
                //         .try_flat_map(|(infer_prev_node, curr_node)| -> Result<_> {
                //             let edges = match curr_node.layer.input_layers() {
                //                 LayerInput::None => vec![],
                //                 LayerInput::Infer => match (infer_prev_node, inputs.len()) {
                //                     // infer as input layer
                //                     (None, 1) => vec![],
                //                     (None, _) => bail!("cannot infer previous layer"),
                //                     (Some(prev_node), _) => vec![(prev_node, curr_node)],
                //                 },
                //                 LayerInput::Single(LayerPath::Layer(from_name))
                //                 | LayerInput::Single(LayerPath::GroupRef {
                //                     layer: from_name,
                //                     ..
                //                 }) => {
                //                     let prev_node = *name_node_map
                //                         .get(from_name)
                //                         .ok_or_else(|| format_err!(""))?;
                //                     vec![(prev_node, curr_node)]
                //                 }
                //                 LayerInput::Multi(from_layers) => {
                //                     let edges: Vec<_> = from_layers
                //                         .iter()
                //                         .cloned()
                //                         .map(|from_name| -> Result<_> {
                //                             let edge = match from_name {
                //                                 LayerPath::Layer(from_name)
                //                                 | LayerPath::GroupRef {
                //                                     layer: from_name, ..
                //                                 } => {
                //                                     let prev_node =
                //                                         *name_node_map
                //                                             .get(from_name)
                //                                             .ok_or_else(|| format_err!(""))?;
                //                                     (prev_node, curr_node)
                //                                 }
                //                             };
                //                             Ok(edge)
                //                         })
                //                         .try_collect()?;
                //                     edges
                //                 }
                //             };
                //             Ok(edges.into_iter().map(Result::Ok))
                //         })
                //         .try_collect()?;

                //     // create graph
                //     let graph = {
                //         let mut graph = DiGraphMap::new();
                //         nodes.into_iter().for_each(|node| {
                //             graph.add_node(node);
                //         });
                //         edges.into_iter().for_each(|(from, to)| {
                //             graph.add_edge(from, to, ());
                //         });
                //         graph
                //     };

                //     let sorted_layers: Vec<_> = petgraph::algo::toposort(&graph, None)
                //         .map_err(|_cycle| format_err!("cycle detected"))?
                //         .into_iter()
                //         .map(
                //             |IndexedLayer {
                //                  layer_ident, layer, ..
                //              }| (layer_ident, layer),
                //         )
                //         .collect();

                //     sorted_layers
                // };

                // outputs.iter().map(|(layer_name, layer)| {
                //     let layer_path: NodePath = prefix.join(layer_name);
                //     (Some(layer_path), layer)
                // });

                // index layers
                // let mut inferred_from_key = if inputs.len() == 1 {
                //     Some(*nodes.keys().next().unwrap())
                // } else {
                //     None
                // };

                // inputs
                //     .iter()
                //     .map(|(layer_name, layer)| (Some(layer_name), Layer::Input(layer)))
                //     .chain(
                //         sorted_layers
                //             .into_iter()
                //             .map(|(layer_ident, layer)| (layer_ident.name(), Layer::Layer(layer))),
                //     )
                //     .chain(outputs.iter().map(|(layer_name, layer)| (Some(layer_name), Layer::Output(layer))))
                //     .try_for_each(|(layer_name, layer)| -> Result<_> {
                //         let (tuples, edges) = match layer {
                //             Layer::Input(_input) => {
                //                 let layer_name = layer_name.unwrap();
                //                 let key = key_enumerator.next().unwrap();
                //                 let path = prefix.join(layer_name);
                //                 (vec![(key, Some(path), layer)], vec![])
                //             }
                //             Layer::Layer(config::Layer::Group(config::Group {
                //                 from,
                //                 group: sub_group_name,
                //             })) => {
                //                 let layer_name = layer_name
                //                     .ok_or_else(|| format_err!("group layer must be named"))?;

                //                 // build subgraph from group
                //                 {
                //                     let new_prefix = prefix.join(layer_name);
                //                     traverse(groups, sub_group_name, &new_prefix, global_nodes, global_edges, key_enumerator, path_key_map)?
                //                 };

                //                 // build edges connecting into the subgraph
                //                 let edges: Vec<_> = from
                //                     .iter()
                //                     .map(|(to_name, from_path)| -> Result<_> {
                //                         let edge = match from_path {
                //                             LayerPath::Layer(from_name) => {
                //                                 let from_path = prefix.join(from_name);
                //                                 let from_key = *path_key_map.get(&from_path).ok_or_else(|| format_err!("the referred layer with name '{}' does not exist", from_name))?;
                //                                 let to_path = prefix.join(layer_name).join(to_name);
                //                                 let to_key = *path_key_map.get(&to_path).ok_or_else(|| format_err!("the referred layer with name '{}' does not exist", to_name))?;
                //                                 (from_key, to_key)
                //                             }
                //                             LayerPath::GroupRef {
                //                                 layer: from_name,
                //                                 output: from_output,
                //                             } => {
                //                                 let from_path = prefix.join(from_name).join(from_output);
                //                                 let from_key = *path_key_map.get(&from_path).ok_or_else(|| format_err!("the referred layer with name '{}' or its output '{}' does not exist", from_name, from_output))?;
                //                                 let to_path = prefix.join(layer_name).join(to_name);
                //                                 let to_key = *path_key_map.get(&to_path).ok_or_else(|| format_err!("the referred layer with name '{}' does not exist", to_name))?;
                //                                 (from_key, to_key)
                //                             }
                //                         };
                //                         Ok(edge)
                //                     })
                //                     .try_collect()?;

                //                 (vec![], edges)
                //             }
                //             Layer::Layer(_) => {
                //                 // create new index for the layer
                //                 let key = key_enumerator.next().unwrap();

                //                 // save path
                //                 let path = layer_name.map(|name| {
                //                     prefix.join(name)
                //                 });

                //                 // add edges
                //                 let edges = match layer.input_layers() {
                //                     LayerInput::None => vec![],
                //                     LayerInput::Infer => {
                //                         todo!();
                //                     }
                //                     LayerInput::Single(LayerPath::Layer(from_name)) => {
                //                         let from_path = prefix.join(from_name);
                //                         let from_key = *path_key_map.get(&from_path).ok_or_else(|| format_err!("the layer with name '{}' does not exist", from_name))?;
                //                         let edge = (from_key, key);
                //                         vec![edge]
                //                     }
                //                     LayerInput::Single(LayerPath::GroupRef{ layer: from_name, output: from_output }) => {
                //                         let from_path = prefix.join(from_name).join(from_output);
                //                         let from_key = *path_key_map.get(&from_path).ok_or_else(|| format_err!("the layer with name '{}' or group output '{}' does not exist", from_name, from_output))?;
                //                         let edge = (from_key, key);
                //                         vec![edge]
                //                     }
                //                     LayerInput::Multi(from_paths) => {
                //                         let edges: Vec<_> = from_paths
                //                             .into_iter()
                //                             .map(|from_path| -> Result<_> {
                //                                 let edge = match from_path {
                //                                     LayerPath::Layer(from_name) => {
                //                                         let from_path = prefix.join(from_name);
                //                                         let from_key = *path_key_map.get(&from_path).ok_or_else(|| format_err!("the layer with name '{}' does not exist", from_name))?;
                //                                         let edge = (from_key, key);
                //                                         edge
                //                                     }
                //                                     LayerPath::GroupRef { layer: from_name, output: from_output } => {
                //                                         let from_path = prefix.join(from_name).join(from_output);
                //                                         let from_key = *path_key_map.get(&from_path).ok_or_else(|| format_err!("the layer with name '{}' or group output '{}' does not exist", from_name, from_output))?;
                //                                         let edge = (from_key, key);
                //                                         edge
                //                                     }
                //                                 };
                //                                 Ok(edge)
                //                             })
                //                             .try_collect()?;
                //                         edges
                //                     }
                //                 };

                //                 (vec![(key, path, layer)], edges)
                //             }
                //             Layer::Output(config::Output { from }) => {
                //                 let layer_name = layer_name.unwrap();
                //                 let path = prefix.join(layer_name);
                //                 let key = key_enumerator.next().unwrap();

                //                 let edge = match from {
                //                     None => {
                //                         // inferred
                //                         todo!();
                //                     }
                //                     Some(LayerPath::Layer(from_name)) => {
                //                         let from_path = prefix.join(from_name);
                //                         let from_key = *path_key_map
                //                             .get(&from_path)
                //                             .ok_or_else(|| format_err!("the layer with name '{}' does not exist", from_name))?;
                //                         (from_key, key)
                //                     }
                //                     Some(LayerPath::GroupRef { layer: from_name, output: from_output }) => {
                //                         let from_path = prefix.join(from_name).join(from_output);
                //                         let from_key = *path_key_map
                //                             .get(&from_path)
                //                             .ok_or_else(|| format_err!("the layer with name '{}' or group output '{}' does not exist", from_name, from_output))?;
                //                         (from_key, key)
                //                     }
                //                 };

                //                 (vec![(key, Some(path), layer)], vec![edge])
                //             }
                //         };

                //         tuples.into_iter().for_each(|(key, path, layer)| {
                //             global_nodes.insert(key, layer);
                //             if let Some(path) = path {
                //                 path_key_map.insert(path, key);
                //             }
                //         });

                //         edges.into_iter().for_each(|(from_key, to_key)| {
                //             global_edges.insert((from_key, to_key));
                //         });

                //         Ok(())
                //     })?;

                Ok(())
            }

            let (nodes, edges) = {
                let mut nodes = HashMap::new();
                let mut edges = HashSet::new();

                traverse(
                    groups,
                    main_group_name,
                    &NodePath::empty(),
                    &mut nodes,
                    &mut edges,
                    &mut iter::repeat(())
                        .enumerate()
                        .map(|(index, ())| NodeKey(index)),
                    &mut HashMap::new(),
                )?;
                (nodes, edges)
            };
            todo!();
        }
    }

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
