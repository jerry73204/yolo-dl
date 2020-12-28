use crate::{
    common::*,
    config::{
        self, Group, GroupName, GroupPath, Groups, Model, Module, ModuleEx, ModuleInput,
        ModuleName, ModulePath,
    },
    utils::{self, IteratorEx},
};

pub use graph::*;
// pub use node::*;

mod graph {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    pub struct Graph {
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<NodeKey, Node, _>"))]
        pub nodes: IndexMap<NodeKey, Node>,
    }

    impl Graph {
        pub fn load(config: &Model) -> Result<Self> {
            let Model { main_group, groups } = config;
            let graph = Self::from_model_groups(&groups, &main_group)?;
            Ok(graph)
        }

        fn from_model_groups(groups: &Groups, main_group_name: &GroupName) -> Result<Self> {
            #[derive(Debug, Clone, PartialEq, Eq, Hash)]
            struct NodeEntry<'a> {
                key: NodeKey,
                path: Option<ModulePath>,
                layer: &'a Module,
            };

            // many-to-one edge
            #[derive(Debug, Clone, PartialEq, Eq, Hash)]
            struct EdgeEntry {
                src: Vec<NodePath>,
                dst: NodePath,
            }

            #[derive(Debug, Clone, PartialEq, Eq, Hash)]
            enum NodePath {
                Resolved(NodeKey),
                Unresolved(ModulePath),
            }

            impl From<NodeKey> for NodePath {
                fn from(from: NodeKey) -> Self {
                    Self::Resolved(from)
                }
            }

            impl From<ModulePath> for NodePath {
                fn from(from: ModulePath) -> Self {
                    Self::Unresolved(from)
                }
            }

            fn traverse_nodes<'a>(
                groups: &'a Groups,
                main_group_name: &GroupName,
                prefix: ModulePath,
                key_enumerator: &mut impl Iterator<Item = NodeKey>,
            ) -> Result<(Vec<NodeEntry<'a>>, Vec<EdgeEntry>)> {
                let group = groups
                    .groups()
                    .get(main_group_name)
                    .ok_or_else(|| format_err!("the group '{}' does not exist", main_group_name))?;

                let mut saved_prev_key = None;
                let tuples: Vec<_> = group
                    .layers()
                    .iter()
                    .map(move |layer: &Module| -> Result<_> {
                        let (nodes, edges) = match layer {
                            Module::GroupRef(config::GroupRef {
                                name: layer_name,
                                group: sub_group_name,
                                from,
                                ..
                            }) => {
                                // enumerate nodes from the group
                                let group_prefix = prefix.join(layer_name);
                                let (nodes, group_edges) = traverse_nodes(
                                    groups,
                                    &sub_group_name,
                                    group_prefix.clone(),
                                    key_enumerator,
                                )?;

                                // create edges
                                let edges: Vec<_> = from
                                    .iter()
                                    .map(|(dst_name, src_path)| -> Result<_> {
                                        // forbid null- or self-reference
                                        {
                                            let first = src_path
                                                .as_ref()
                                                .first()
                                                .ok_or_else(|| format_err!("TODO"))?;
                                            ensure!(first != layer_name, "TODO");
                                        }

                                        let src = prefix.extend(src_path);
                                        let dst = group_prefix.join(dst_name);
                                        Ok(EdgeEntry {
                                            src: vec![src.into()],
                                            dst: dst.into(),
                                        })
                                    })
                                    .chain(group_edges.into_iter().map(Ok))
                                    .try_collect()?;

                                saved_prev_key = None;

                                (nodes, edges)
                            }
                            layer => {
                                // create key and path for the layer
                                let key = key_enumerator.next().unwrap();
                                let path = layer.name().map(|name| prefix.join(name));
                                let infer_prev_key = saved_prev_key;
                                saved_prev_key = Some(key);

                                let nodes = vec![NodeEntry { key, path, layer }];

                                // create edges
                                let edges = match layer.input_paths() {
                                    ModuleInput::None => vec![],
                                    ModuleInput::Infer => {
                                        let prev_key =
                                            infer_prev_key.ok_or_else(|| format_err!("TODO"))?;
                                        vec![EdgeEntry {
                                            src: vec![prev_key.into()],
                                            dst: key.into(),
                                        }]
                                    }
                                    ModuleInput::Path(from_path) => {
                                        let src = prefix.extend(from_path);
                                        vec![EdgeEntry {
                                            src: vec![src.into()],
                                            dst: key.into(),
                                        }]
                                    }
                                    ModuleInput::Indexed(from_paths) => {
                                        let src: Vec<NodePath> = from_paths
                                            .iter()
                                            .cloned()
                                            .map(|path| path.into())
                                            .collect();
                                        vec![EdgeEntry {
                                            src,
                                            dst: key.into(),
                                        }]
                                    }
                                    ModuleInput::Named(_from_paths) => {
                                        unreachable!("please report bug");
                                    }
                                };

                                (nodes, edges)
                            }
                        };

                        Ok((nodes, edges))
                    })
                    .try_collect()?;
                let (nodes_vec, edges_vec) = tuples.into_iter().unzip_n_vec();
                let nodes: Vec<_> = nodes_vec.into_iter().flatten().collect();
                let edges: Vec<_> = edges_vec.into_iter().flatten().collect();

                Ok((nodes, edges))
            }

            // collect nodes and edges from groups
            let (node_entires, edges_entries) = traverse_nodes(
                groups,
                main_group_name,
                ModulePath::empty(),
                &mut iter::repeat(())
                    .enumerate()
                    .map(|(index, ())| NodeKey(index)),
            )?;

            let path_key_map: HashMap<_, _> = node_entires
                .iter()
                .flat_map(|node| Some((node.path.clone()?, node.key)))
                .collect();

            let nodes: HashMap<_, _> = node_entires
                .into_iter()
                .map(|node| (node.key, node))
                .collect();

            let dst_src_map = edges_entries
                .into_iter()
                .map(|edge| -> Result<_> {
                    let EdgeEntry { src, dst } = edge;
                    let src: Vec<_> = src
                        .into_iter()
                        .map(|src| -> Result<_> {
                            let key = match src {
                                NodePath::Resolved(key) => key,
                                NodePath::Unresolved(path) => {
                                    *path_key_map.get(&path).ok_or_else(|| format_err!("TODO"))?
                                }
                            };
                            Ok(key)
                        })
                        .try_collect()?;
                    let dst = match dst {
                        NodePath::Resolved(key) => key,
                        NodePath::Unresolved(path) => {
                            *path_key_map.get(&path).ok_or_else(|| format_err!("TODO"))?
                        }
                    };
                    Ok((src, dst))
                })
                .try_fold(HashMap::new(), |mut map, result| -> Result<_> {
                    let (src, dst) = result?;
                    let prev = map.insert(dst, src);
                    ensure!(prev.is_none(), "TODO");
                    Ok(map)
                })?;

            // sanity check
            nodes.values().try_for_each(|node| -> Result<_> {
                let NodeEntry {
                    key,
                    ref path,
                    layer,
                } = *node;

                match layer {
                    Module::Input(_) => {
                        let path = path.as_ref().unwrap();
                        let src = dst_src_map.get(&key);

                        // check if
                        // (1) top-level input layer has no incoming edge
                        // (2) non-top level input has an incoming edge
                        match (path.depth(), src) {
                            (1, Some(_)) => {
                                bail!("top level input layer cannot have an incoming edge")
                            }
                            (1, None) | (_, Some(_)) => {}
                            (_, None) => {
                                bail!("non-top level input layer must have an incoming edge")
                            }
                        }
                    }
                    Module::GroupRef(_) => unreachable!("please report bug"),
                    layer => {
                        let src = dst_src_map.get(&key);

                        // make sure the input satisfies the specification
                        match layer.input_paths() {
                            ModuleInput::None => ensure!(src.is_none(), "TODO"),
                            ModuleInput::Infer | ModuleInput::Path(_) => {
                                ensure!(src.map(|keys| keys.len() == 1).unwrap_or(false), "TODO")
                            }
                            ModuleInput::Indexed(from_paths) => ensure!(
                                src.map(|keys| keys.len() == from_paths.len())
                                    .unwrap_or(false),
                                "TODO"
                            ),
                            ModuleInput::Named(_) => unreachable!(),
                        }
                    }
                }

                Ok(())
            })?;

            // toposort
            let sorted_node_keys = {
                let mut graph = DiGraphMap::new();
                nodes.values().for_each(|node| {
                    let NodeEntry { key, .. } = *node;
                    graph.add_node(key);

                    let src = dst_src_map.get(&key);
                    src.iter()
                        .flat_map(|src_keys| src_keys.iter())
                        .for_each(|&src_key| {
                            graph.add_edge(src_key, key, ());
                        });
                });

                let sorted_nodes: Vec<NodeKey> = petgraph::algo::toposort(&graph, None)
                    .map_err(|_| format_err!("cycle detected"))?;
                sorted_nodes
            };

            todo!();
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Node {
        pub key: NodeKey,
        pub edges: Vec<IncomingEdge>,
    }

    // #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    // pub struct NodeInput {
    //     pub src: Vec<IncomingEdge>,
    //     pub dst: NodeKey,
    // }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct IncomingEdge {
        pub src: NodeKey,
        pub shape: Vec<usize>,
    }

    #[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct NodeKey(pub usize);
}

// mod node {
//     use super::*;
// }
