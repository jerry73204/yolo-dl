use crate::common::*;

pub trait NodeId
where
    Self: Copy + Ord + Eq + Hash + Debug,
{
}

impl<T> NodeId for T where T: Copy + Ord + Eq + Hash + Debug {}

#[derive(Debug, Clone)]
pub struct MultiDag<N, NodeData, EdgeData>
where
    N: NodeId,
{
    nodes: IndexMap<N, Node<N, NodeData, EdgeData>>,
}

impl<N, NodeData, EdgeData> MultiDag<N, NodeData, EdgeData>
where
    N: NodeId,
{
    pub fn into_iter(self) -> impl Iterator<Item = (N, Node<N, NodeData, EdgeData>)> {
        self.nodes.into_iter()
    }

    pub fn iter(&self) -> impl Iterator<Item = (N, &'_ Node<N, NodeData, EdgeData>)> {
        self.nodes.iter().map(|(&id, node)| (id, node))
    }
}

impl<N, NodeData, EdgeData> FromIterator<Node<N, NodeData, EdgeData>>
    for Result<MultiDag<N, NodeData, EdgeData>>
where
    N: NodeId,
{
    fn from_iter<T: IntoIterator<Item = Node<N, NodeData, EdgeData>>>(iter: T) -> Self {
        // collect nodes
        let nodes: HashMap<_, _> = iter.into_iter().map(|node| (node.id, node)).try_fold(
            HashMap::new(),
            |mut map, (id, node)| {
                let prev = map.insert(id, node);
                ensure!(prev.is_none(), "duplicated node id {:?}", id);
                Ok(map)
            },
        )?;

        // validate edges
        nodes.iter().try_for_each(|(id, node)| {
            node.src.iter().try_for_each(|(src_id, _edge)| {
                ensure!(
                    nodes.get(src_id).is_some(),
                    "node {:?} has an nonexist source node {:?}",
                    id,
                    src_id
                );
                Ok(())
            })
        })?;

        // toposort nodes
        let sorted_ids = {
            let graph = nodes
                .iter()
                .fold(DiGraphMap::new(), |mut graph, (&id, node)| {
                    graph.add_node(id);
                    node.src.iter().for_each(|(src_id, _edge)| {
                        graph.add_edge(*src_id, id, ());
                    });
                    graph
                });
            let sorted_ids = petgraph::algo::toposort(&graph, None)
                .map_err(|cycle| format_err!("cycle detected at node {:?}", cycle.node_id()))?;
            sorted_ids
        };

        let nodes: IndexMap<_, _> = {
            let mut orig_nodes = nodes;
            sorted_ids
                .into_iter()
                .map(|id| {
                    let node = orig_nodes.remove(&id).unwrap();
                    (id, node)
                })
                .collect()
        };

        Ok(MultiDag { nodes })
    }
}

impl<N, NodeData, EdgeData> Index<&'_ N> for MultiDag<N, NodeData, EdgeData>
where
    N: NodeId,
{
    type Output = Node<N, NodeData, EdgeData>;

    fn index(&self, id: &'_ N) -> &Self::Output {
        &self.nodes[id]
    }
}

impl<N, NodeData, EdgeData> Index<N> for MultiDag<N, NodeData, EdgeData>
where
    N: NodeId,
{
    type Output = Node<N, NodeData, EdgeData>;

    fn index(&self, id: N) -> &Self::Output {
        &self.nodes[&id]
    }
}

#[derive(Debug, Clone)]
pub struct Node<N, NodeData, EdgeData>
where
    N: NodeId,
{
    pub id: N,
    pub data: NodeData,
    pub src: Vec<(N, EdgeData)>,
}
