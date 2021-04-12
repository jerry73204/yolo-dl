use super::module::{Input, Module, ModuleInput, ModuleOutput};
use crate::common::*;
use tch_goodies::module::{
    Concat2D, ConvBn2DInit, DarkBatchNormInit, DarkCsp2DInit, DeconvBn2DInit, Detect2DInit,
    MergeDetect2D, SppCsp2DInit, Sum2D, UpSample2D,
};

pub use model_config::graph::{InputKeys, NodeKey};
pub use yolo_model::*;

mod yolo_model {
    use super::*;

    /// The model running on libtorch.
    #[derive(Debug)]
    pub struct YoloModel {
        pub(crate) layers: IndexMap<NodeKey, Layer>,
        pub(crate) output_key: NodeKey,
    }

    impl YoloModel {
        /// Run forward pass.
        pub fn forward_t(&mut self, input: &Tensor, train: bool) -> Result<ModuleOutput> {
            let Self {
                ref mut layers,
                output_key,
            } = *self;
            let mut module_outputs: HashMap<NodeKey, ModuleOutput> = HashMap::new();
            let mut input = Some(input); // it makes sure the input is consumed at most once

            // run the network
            layers.values_mut().try_for_each(|layer| -> Result<_> {
                let Layer {
                    key,
                    ref mut module,
                    ref input_keys,
                } = *layer;

                let module_input: ModuleInput = match input_keys {
                    InputKeys::None => ModuleInput::None,
                    InputKeys::PlaceHolder => input.take().unwrap().into(),
                    InputKeys::Single(src_key) => (&module_outputs[src_key]).try_into()?,
                    InputKeys::Indexed(src_keys) => {
                        let inputs: Vec<_> = src_keys
                            .iter()
                            .map(|src_key| &module_outputs[src_key])
                            .collect();
                        inputs.as_slice().try_into()?
                    }
                };

                let module_output = module
                    .forward_t(module_input, train)
                    .with_context(|| format!("forward error at node {}", key))?;
                module_outputs.insert(key, module_output);

                Ok(())
            })?;

            debug_assert!(input.is_none());

            // extract output
            let output = module_outputs.remove(&output_key).unwrap();

            Ok(output)
        }

        /// Build a model from a computation graph.
        pub fn from_graph<'p>(
            path: impl Borrow<nn::Path<'p>>,
            orig_graph: &'_ model_config::graph::Graph,
        ) -> Result<Self> {
            use model_config::{config, graph};
            let path = path.borrow();
            let orig_nodes = orig_graph.nodes();

            let layers: IndexMap<_, _> = orig_nodes
                .iter()
                .map(|(&key, node)| -> Result<_> {
                    let graph::Node {
                        input_keys, config, ..
                    } = node;
                    let module_path = path / format!("module_{}", key);

                    let module = match *config {
                        config::Module::Input(config::Input { ref shape, .. }) => {
                            Module::Input(Input::new(shape))
                        }
                        config::Module::ConvBn2D(config::ConvBn2D {
                            c,
                            k,
                            s,
                            p,
                            d,
                            g,
                            act,
                            bias,
                            ref bn,
                            ..
                        }) => {
                            let src_key = input_keys.single().unwrap();
                            let [_b, in_c, _h, _w] = orig_nodes[&src_key]
                                .output_shape
                                .tensor()
                                .unwrap()
                                .size4()
                                .unwrap();
                            let in_c = in_c.size().unwrap();

                            Module::ConvBn2D(
                                ConvBn2DInit {
                                    in_c,
                                    out_c: c,
                                    k,
                                    s,
                                    p,
                                    d,
                                    g,
                                    bias,
                                    activation: act,
                                    batch_norm: bn.enabled.then(|| {
                                        let mut config = DarkBatchNormInit::default();
                                        if !bn.affine {
                                            config.ws_init = None;
                                            config.bs_init = None;
                                        }
                                        config
                                    }),
                                }
                                .build(module_path),
                            )
                        }
                        config::Module::DeconvBn2D(config::DeconvBn2D {
                            c,
                            k,
                            s,
                            p,
                            op,
                            d,
                            g,
                            bias,
                            act,
                            ref bn,
                            ..
                        }) => {
                            let src_key = input_keys.single().unwrap();
                            let [_b, in_c, _h, _w] = orig_nodes[&src_key]
                                .output_shape
                                .tensor()
                                .unwrap()
                                .size4()
                                .unwrap();
                            let in_c = in_c.size().unwrap();

                            Module::DeconvBn2D(
                                DeconvBn2DInit {
                                    in_c,
                                    out_c: c,
                                    k,
                                    s,
                                    p,
                                    op,
                                    d,
                                    g,
                                    bias,
                                    activation: act,
                                    batch_norm: bn.enabled.then(|| {
                                        let mut config = DarkBatchNormInit::default();
                                        if !bn.affine {
                                            config.ws_init = None;
                                            config.bs_init = None;
                                        }
                                        config
                                    }),
                                }
                                .build(module_path),
                            )
                        }
                        config::Module::UpSample2D(config::UpSample2D { scale, .. }) => {
                            Module::UpSample2D(UpSample2D::new(scale.raw())?)
                        }
                        config::Module::DarkCsp2D(config::DarkCsp2D {
                            c,
                            repeat,
                            shortcut,
                            c_mul,
                            ref bn,
                            ..
                        }) => {
                            let src_key = input_keys.single().unwrap();
                            let [_b, in_c, _h, _w] = orig_nodes[&src_key]
                                .output_shape
                                .tensor()
                                .unwrap()
                                .size4()
                                .unwrap();
                            let in_c = in_c.size().unwrap();

                            Module::DarkCsp2D(
                                DarkCsp2DInit {
                                    in_c,
                                    out_c: c,
                                    repeat,
                                    shortcut,
                                    c_mul,
                                    batch_norm: bn.enabled.then(|| {
                                        let mut config = DarkBatchNormInit::default();
                                        if !bn.affine {
                                            config.ws_init = None;
                                            config.bs_init = None;
                                        }
                                        config
                                    }),
                                }
                                .build(module_path),
                            )
                        }
                        config::Module::SppCsp2D(config::SppCsp2D {
                            c,
                            ref k,
                            c_mul,
                            ref bn,
                            ..
                        }) => {
                            let src_key = input_keys.single().unwrap();
                            let [_b, in_c, _h, _w] = orig_nodes[&src_key]
                                .output_shape
                                .tensor()
                                .unwrap()
                                .size4()
                                .unwrap();
                            let in_c = in_c.size().unwrap();

                            Module::SppCsp2D(
                                SppCsp2DInit {
                                    in_c,
                                    out_c: c,
                                    k: k.to_owned(),
                                    c_mul,
                                    batch_norm: bn.enabled.then(|| {
                                        let mut config = DarkBatchNormInit::default();
                                        if !bn.affine {
                                            config.ws_init = None;
                                            config.bs_init = None;
                                        }
                                        config
                                    }),
                                }
                                .build(module_path),
                            )
                        }
                        config::Module::Sum2D(_) => Module::Sum2D(Sum2D),
                        config::Module::Concat2D(_) => Module::Concat2D(Concat2D),
                        config::Module::Detect2D(config::Detect2D {
                            classes,
                            ref anchors,
                            ..
                        }) => {
                            let anchors: Vec<_> = anchors
                                .iter()
                                .map(|size| -> Result<_> {
                                    let config::Size { h, w } = *size;
                                    let size =
                                        RatioSize::new(h.try_into()?, w.try_into()?).unwrap();
                                    Ok(size)
                                })
                                .try_collect()?;

                            Module::Detect2D(
                                Detect2DInit {
                                    num_classes: classes,
                                    anchors,
                                }
                                .build(module_path),
                            )
                        }
                        config::Module::GroupRef(_) => unreachable!(),
                        config::Module::MergeDetect2D(_) => {
                            Module::MergeDetect2D(MergeDetect2D::new())
                        }
                    };

                    let layer = Layer {
                        key,
                        input_keys: input_keys.to_owned(),
                        module,
                    };

                    Ok((key, layer))
                })
                .try_collect()?;

            let output_key = {
                let mut iter = layers
                    .iter()
                    .filter_map(|(&key, layer)| match layer.module {
                        Module::MergeDetect2D(_) => Some(key),
                        _ => None,
                    });
                let output_key = iter.next().ok_or_else(|| format_err!("TODO"))?;
                ensure!(iter.next() == None, "TODO");
                output_key
            };

            Ok(Self { layers, output_key })
        }
    }
}

#[derive(Debug)]
pub struct Layer {
    pub(crate) key: NodeKey,
    pub(crate) module: Module,
    pub(crate) input_keys: InputKeys,
}
