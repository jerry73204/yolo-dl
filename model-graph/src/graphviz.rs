#![cfg(feature = "dot")]

use super::*;
use crate::{common::*, utils::OptionEx};
use dot::{Arrow, Edges, GraphWalk, Id, LabelText, Labeller, Nodes, Style};
use model_config::{self as config, Module};

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
                let config::ConvBn2D {
                    k, s, p, d, ref bn, ..
                } = *conv;

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
                    if bn.enabled { "yes" } else { "no" }
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
