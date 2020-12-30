use crate::{
    common::*,
    config::{
        self, GroupName, Groups, Model, Module, ModuleEx, ModuleInput, ModulePath, Shape,
        ShapeInput,
    },
    utils::{self, OptionEx},
};

pub use graph::*;

mod graph {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    pub struct Graph {
        #[derivative(Hash(hash_with = "utils::hash_vec_indexmap::<NodeKey, Node, _>"))]
        nodes: IndexMap<NodeKey, Node>,
    }

    impl Graph {
        pub fn new(config: &Model) -> Result<Self> {
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

            #[derive(Debug, Clone, PartialEq, Eq, Hash)]
            enum InputPaths {
                None,
                Single(NodePath),
                Indexed(Vec<NodePath>),
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
            ) -> Result<(Vec<NodeEntry<'a>>, Vec<(NodePath, InputPaths)>)> {
                let group = groups
                    .groups()
                    .get(main_group_name)
                    .ok_or_else(|| format_err!("the group '{}' does not exist", main_group_name))?;

                let mut saved_prev_key = None;
                let tuples: Vec<_> = group
                    .layers()
                    .iter()
                    .map(move |layer: &Module| -> Result<_> {
                        let (nodes, input_paths) = match layer {
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
                                let input_paths: Vec<_> = from
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
                                        Ok((dst.into(), InputPaths::Single(src.into())))
                                    })
                                    .chain(group_edges.into_iter().map(Ok))
                                    .try_collect()?;

                                saved_prev_key = None;

                                (nodes, input_paths)
                            }
                            layer => {
                                // create key and path for the layer
                                let key = key_enumerator.next().unwrap();
                                let path = layer.name().map(|name| prefix.join(name));
                                let infer_prev_key = saved_prev_key;
                                saved_prev_key = Some(key);

                                let node = NodeEntry { key, path, layer };

                                // create input paths
                                let input_paths = match layer.input_paths() {
                                    ModuleInput::None => {
                                        // no input if it's top-level input layer
                                        match (layer, prefix.is_empty()) {
                                            (Module::Input(_), true) => Some(InputPaths::None),
                                            (Module::Input(_), false) => None,
                                            (_, _) => None,
                                        }
                                    }
                                    ModuleInput::Infer => {
                                        let prev_key =
                                            infer_prev_key.ok_or_else(|| format_err!("TODO"))?;
                                        Some(InputPaths::Single(prev_key.into()))
                                    }
                                    ModuleInput::Path(from_path) => {
                                        let src_path = prefix.extend(from_path);
                                        Some(InputPaths::Single(src_path.into()))
                                    }
                                    ModuleInput::Indexed(from_paths) => {
                                        let src_paths: Vec<NodePath> = from_paths
                                            .iter()
                                            .cloned()
                                            .map(|from_path| prefix.extend(&from_path).into())
                                            .collect();
                                        Some(InputPaths::Indexed(src_paths))
                                    }
                                    ModuleInput::Named(_from_paths) => {
                                        unreachable!("please report bug");
                                    }
                                };

                                let input_pairs: Vec<_> = input_paths
                                    .into_iter()
                                    .map(|input_paths| (key.into(), input_paths))
                                    .collect();

                                (vec![node], input_pairs)
                            }
                        };

                        Ok((nodes, input_paths))
                    })
                    .try_collect()?;
                let (nodes_vec, input_paths_vec) = tuples.into_iter().unzip_n_vec();
                let nodes: Vec<_> = nodes_vec.into_iter().flatten().collect();
                let input_paths: Vec<_> = input_paths_vec.into_iter().flatten().collect();

                Ok((nodes, input_paths))
            }

            // collect nodes and edges from groups
            let (node_entires, input_paths) = traverse_nodes(
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

            let input_keys_map: HashMap<NodeKey, InputKeys> = input_paths
                .into_iter()
                .map(|(dst_path, src_path)| -> Result<_> {
                    let dst_key = match dst_path {
                        NodePath::Resolved(key) => key,
                        NodePath::Unresolved(path) => {
                            *path_key_map.get(&path).ok_or_else(|| format_err!("TODO"))?
                        }
                    };

                    let src_keys = match src_path {
                        InputPaths::None => InputKeys::None,
                        InputPaths::Single(src_path) => {
                            let src_key = match src_path {
                                NodePath::Resolved(key) => key,
                                NodePath::Unresolved(path) => *path_key_map
                                    .get(&path)
                                    .ok_or_else(|| format_err!("cannot resolve '{}'", path))?,
                            };

                            InputKeys::Single(src_key)
                        }
                        InputPaths::Indexed(src_paths) => {
                            let src_keys: Vec<_> = src_paths
                                .into_iter()
                                .map(|src| -> Result<_> {
                                    let key = match src {
                                        NodePath::Resolved(key) => key,
                                        NodePath::Unresolved(path) => {
                                            *path_key_map.get(&path).ok_or_else(|| {
                                                format_err!("cannot resolve '{}'", path)
                                            })?
                                        }
                                    };
                                    Ok(key)
                                })
                                .try_collect()?;

                            InputKeys::Indexed(src_keys)
                        }
                    };

                    Ok((dst_key, src_keys))
                })
                .try_fold(HashMap::new(), |mut map, result| -> Result<_> {
                    let (dst_key, src_keys) = result?;
                    let prev = map.insert(dst_key, src_keys);
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
                        let src = &input_keys_map[&key];

                        // check if
                        // (1) top-level input layer has no incoming edge
                        // (2) non-top level input has an incoming edge
                        match (path.depth(), src) {
                            (1, InputKeys::None) => {}
                            (1, _) => {
                                bail!("top level input cannot have inputs")
                            }
                            (_, InputKeys::Single(_)) => {}
                            (_, _) => {
                                bail!("non-top level input must have an input")
                            }
                        }
                    }
                    Module::GroupRef(_) => unreachable!("please report bug"),
                    layer => {
                        let input_keys = &input_keys_map[&key];

                        // make sure the input satisfies the specification
                        match (layer.input_paths(), input_keys) {
                            (ModuleInput::None, InputKeys::None)
                            | (ModuleInput::Infer, InputKeys::Single(_))
                            | (ModuleInput::Path(_), InputKeys::Single(_))
                            | (ModuleInput::Indexed(_), InputKeys::Indexed(_)) => (),
                            (ModuleInput::Named(_), _) => unreachable!("please report bug"),
                            _ => bail!("TODO"),
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

                    let input_keys = &input_keys_map[&key];
                    input_keys.iter().for_each(|src_key| {
                        graph.add_edge(src_key, key, ());
                    });
                });

                let sorted_nodes: Vec<NodeKey> = petgraph::algo::toposort(&graph, None)
                    .map_err(|_| format_err!("cycle detected"))?;

                sorted_nodes
            };

            // compute output shape
            let output_shape_map: HashMap<NodeKey, Shape> =
                sorted_node_keys.iter().cloned().try_fold(
                    HashMap::new(),
                    |mut output_shape_map: HashMap<NodeKey, Shape>, key| -> Result<_> {
                        let NodeEntry {
                            layer, ref path, ..
                        } = nodes[&key];
                        let input_keys = &input_keys_map[&key];

                        let output_shape = match layer {
                            Module::Input(config::Input { shape, .. }) => {
                                let path = path.as_ref().unwrap();

                                if path.depth() == 1 {
                                    // top level input has no input
                                    ensure!(matches!(input_keys, InputKeys::None), "TODO");
                                    let output_shape = shape.to_owned();
                                    output_shape
                                } else {
                                    // non-top level input has one input
                                    let src_key = match input_keys {
                                        InputKeys::Single(key) => key,
                                        _ => bail!("TODO"),
                                    };
                                    let input_shape = &output_shape_map[&src_key];

                                    // ensure input shape is consistent with specified shape
                                    let output_shape = input_shape
                                        .equalize(shape)
                                        .ok_or_else(|| format_err!("TODO"))?;

                                    output_shape
                                }
                            }
                            Module::GroupRef(_) => unreachable!(),
                            layer => match input_keys {
                                InputKeys::None => {
                                    let input_shape = ShapeInput::None;
                                    let output_shape = layer
                                        .output_shape(input_shape)
                                        .ok_or_else(|| format_err!("TODO"))?;
                                    output_shape
                                }
                                InputKeys::PlaceHolder => {
                                    let input_shape = ShapeInput::None;
                                    let output_shape = layer
                                        .output_shape(input_shape)
                                        .ok_or_else(|| format_err!("TODO"))?;
                                    output_shape
                                }
                                InputKeys::Single(src_key) => {
                                    let input_shape = &output_shape_map[&src_key];
                                    let input_shape = ShapeInput::Single(input_shape);
                                    let output_shape = layer
                                        .output_shape(input_shape)
                                        .ok_or_else(|| format_err!("TODO"))?;
                                    output_shape
                                }
                                InputKeys::Indexed(src_keys) => {
                                    let input_shape: Vec<_> = src_keys
                                        .iter()
                                        .cloned()
                                        .map(|src_key| &output_shape_map[&src_key])
                                        .collect();
                                    let input_shape = ShapeInput::Indexed(&input_shape);
                                    let output_shape = layer
                                        .output_shape(input_shape)
                                        .ok_or_else(|| format_err!("TODO"))?;
                                    output_shape
                                }
                            },
                        };

                        output_shape_map.insert(key, output_shape);
                        Ok(output_shape_map)
                    },
                )?;

            // aggregate computed items
            let nodes = {
                let mut output_shape_map = output_shape_map;
                let mut input_keys_map = input_keys_map;
                let mut nodes = nodes;

                let nodes: IndexMap<_, _> = sorted_node_keys
                    .into_iter()
                    .map(|key| {
                        let output_shape = output_shape_map.remove(&key).unwrap();
                        let input_keys = input_keys_map.remove(&key).unwrap();
                        let NodeEntry {
                            layer: config,
                            path,
                            ..
                        } = nodes.remove(&key).unwrap();
                        let node = Node {
                            input_keys,
                            output_shape,
                            path,
                            config: config.to_owned(),
                        };
                        (key, node)
                    })
                    .collect();

                nodes
            };

            let graph = Graph { nodes };
            Ok(graph)
        }

        pub fn nodes(&self) -> &IndexMap<NodeKey, Node> {
            &self.nodes
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Node {
        pub input_keys: InputKeys,
        pub output_shape: Shape,
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
}

// graphviz support

#[cfg(feature = "dot")]
mod graphviz {
    use super::*;
    use dot::{Arrow, Edges, GraphWalk, Id, LabelText, Labeller, Nodes, Style};

    impl Graph {
        pub fn render_dot(&self, writer: &mut impl Write) -> Result<()> {
            dot::render(self, writer)?;
            Ok(())
        }
    }

    impl<'a> GraphWalk<'a, NodeKey, (NodeKey, NodeKey)> for Graph {
        fn nodes(&'a self) -> Nodes<'a, NodeKey> {
            let keys: Vec<_> = self.nodes().keys().cloned().collect();
            keys.into()
        }

        fn edges(&'a self) -> Edges<'a, (NodeKey, NodeKey)> {
            let edges: Vec<_> = self
                .nodes()
                .iter()
                .flat_map(|(&dst_key, node)| {
                    node.input_keys
                        .iter()
                        .map(move |src_key| (src_key, dst_key))
                })
                .collect();
            edges.into()
        }

        fn source(&'a self, edge: &(NodeKey, NodeKey)) -> NodeKey {
            let (src, _dst) = *edge;
            src
        }

        fn target(&'a self, edge: &(NodeKey, NodeKey)) -> NodeKey {
            let (_src, dst) = *edge;
            dst
        }
    }

    impl<'a> Labeller<'a, NodeKey, (NodeKey, NodeKey)> for Graph {
        fn graph_id(&'a self) -> Id<'a> {
            Id::new("darknet").unwrap()
        }

        fn node_id(&'a self, key: &NodeKey) -> Id<'a> {
            Id::new(format!("node_{}", key)).unwrap()
        }

        fn node_shape(&'a self, key: &NodeKey) -> Option<LabelText<'a>> {
            match self.nodes()[key].config {
                Module::Input(_) | Module::Detect2D(_) => Some(LabelText::label("box")),
                Module::Concat2D(_) => Some(LabelText::label("invhouse")),
                Module::Sum2D(_) => Some(LabelText::label("invtrapezium")),
                _ => None,
            }
        }

        fn node_label(&'a self, key: &NodeKey) -> LabelText<'a> {
            let Node {
                config,
                output_shape,
                path,
                ..
            } = &self.nodes()[key];

            match config {
                Module::Input(_) => LabelText::escaped(format!(
                    r"input\n{}",
                    dot::escape_html(&format!("{}", output_shape))
                )),
                Module::ConvBn2D(conv) => {
                    let config::ConvBn2D { k, s, p, d, bn, .. } = *conv;

                    LabelText::escaped(format!(
                        "({}) {}
{}
{}\
k={} s={} p={} d={}
batch_norm={}",
                        key,
                        config.as_ref(),
                        dot::escape_html(&output_shape.to_string()),
                        path.as_ref()
                            .map(|path| format!("path={}\n", path))
                            .display(),
                        k,
                        s,
                        p,
                        d,
                        if bn { "yes" } else { "no" }
                    ))
                }
                Module::DarkCsp2D(csp) => {
                    let config::DarkCsp2D {
                        c,
                        repeat,
                        shortcut,
                        c_mul,
                        ..
                    } = *csp;

                    LabelText::escaped(format!(
                        "({}) {}
{}
{}\
c={} c_mul={}
repeat={}
shortcut={}",
                        key,
                        config.as_ref(),
                        dot::escape_html(&output_shape.to_string()),
                        path.as_ref()
                            .map(|path| format!("path={}\n", path))
                            .display(),
                        c,
                        c_mul,
                        repeat,
                        if shortcut { "yes" } else { "no" },
                    ))
                }
                Module::SppCsp2D(spp) => {
                    let config::SppCsp2D { c, ref k, .. } = *spp;

                    LabelText::escaped(format!(
                        "({}) {}
{}
{}\
c={}
k={:?}",
                        key,
                        config.as_ref(),
                        dot::escape_html(&output_shape.to_string()),
                        path.as_ref()
                            .map(|path| format!("path={}\n", path))
                            .display(),
                        c,
                        k
                    ))
                }
                // Node::MaxPool(node) => {
                //     let MaxPoolNode {
                //         config:
                //             MaxPoolConfig {
                //                 size,
                //                 stride_x,
                //                 stride_y,
                //                 padding,
                //                 ..
                //             },
                //         ..
                //     } = *node;

                //     if stride_y == stride_x {
                //         LabelText::escaped(format!(
                //             r"({}) {}\n{}\nk={} s={} p={}",
                //             layer_index,
                //             self.layers[layer_index].as_ref(),
                //             dot::escape_html(&format!("{:?}", output_shape)),
                //             size,
                //             stride_y,
                //             padding,
                //         ))
                //     } else {
                //         LabelText::escaped(format!(
                //             r"({}) {}\n{}\nk={} sy={} sx={} p={}",
                //             layer_index,
                //             self.layers[layer_index].as_ref(),
                //             dot::escape_html(&format!("{:?}", output_shape)),
                //             size,
                //             stride_y,
                //             stride_x,
                //             padding,
                //         ))
                //     }
                // }
                _ => LabelText::escaped(format!(
                    r"({}) {}
{}
{}",
                    key,
                    config.as_ref(),
                    dot::escape_html(&output_shape.to_string()),
                    path.as_ref()
                        .map(|path| format!("path={}\n", path))
                        .display()
                )),
            }
        }

        fn node_style(&'a self, _node: &NodeKey) -> Style {
            Style::None
        }

        fn node_color(&'a self, key: &NodeKey) -> Option<LabelText<'a>> {
            match self.nodes()[key].config {
                Module::Input(_) => Some(LabelText::label("black")),
                Module::Detect2D(_) => Some(LabelText::label("orange")),
                Module::ConvBn2D(_) => Some(LabelText::label("blue")),
                // Module::MaxPool(_) => Some(LabelText::label("green")),
                Module::Sum2D(_) => Some(LabelText::label("brown")),
                Module::Concat2D(_) => Some(LabelText::label("brown")),
                _ => None,
            }
        }

        fn edge_label(&'a self, edge: &(NodeKey, NodeKey)) -> LabelText<'a> {
            let (src_key, dst_key) = *edge;
            let shape = &self.nodes()[&src_key].output_shape;

            LabelText::escaped(format!(
                r"{} -> {}\n{}",
                src_key,
                dst_key,
                dot::escape_html(&shape.to_string())
            ))
        }

        fn edge_start_arrow(&'a self, _edge: &(NodeKey, NodeKey)) -> Arrow {
            Arrow::none()
        }

        fn edge_end_arrow(&'a self, _edge: &(NodeKey, NodeKey)) -> Arrow {
            Arrow::normal()
        }

        fn edge_style(&'a self, _node: &(NodeKey, NodeKey)) -> Style {
            Style::None
        }

        fn edge_color(&'a self, _node: &(NodeKey, NodeKey)) -> Option<LabelText<'a>> {
            None
        }

        fn kind(&self) -> dot::Kind {
            dot::Kind::Digraph
        }
    }
}
