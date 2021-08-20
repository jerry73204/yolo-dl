use crate::common::*;
use model_config as config;
use model_graph as graph;
use tch_modules as modules;

/// The model running on libtorch.
#[derive(Debug)]
pub struct YoloModel {
    pub(crate) layers: IndexMap<graph::NodeKey, Layer>,
    pub(crate) output_key: graph::NodeKey,
}

impl YoloModel {
    pub fn load_newslab_v1_json<'p>(
        path: impl Borrow<nn::Path<'p>>,
        cfg_file: impl AsRef<Path>,
    ) -> Result<Self> {
        let graph = graph::Graph::load_newslab_v1_json(cfg_file)?;
        let model = Self::from_graph(path, &graph)?;
        Ok(model)
    }

    /// Build a model from a computation graph.
    pub fn from_graph<'p>(
        path: impl Borrow<nn::Path<'p>>,
        orig_graph: &graph::Graph,
    ) -> Result<Self> {
        let path = path.borrow();
        let orig_nodes = orig_graph.nodes();

        let layers: IndexMap<_, _> = orig_nodes
            .iter()
            .map(|(&key, node)| -> Result<_> {
                let graph::Node {
                    input_keys, config, ..
                } = node;
                let module_path = path / format!("module_{}", key);

                let module: modules::Module = match *config {
                    config::Module::Conv2D(config::Conv2D {
                        c: out_c,
                        k,
                        s,
                        p,
                        d,
                        g,
                        bias,
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

                        nn::conv2d(
                            module_path,
                            in_c as i64,
                            out_c as i64,
                            k as i64,
                            nn::ConvConfig {
                                stride: s as i64,
                                padding: p as i64,
                                dilation: d as i64,
                                groups: g as i64,
                                bias,
                                ..Default::default()
                            },
                        )
                        .into()
                    }
                    config::Module::Input(config::Input { ref shape, .. }) => {
                        modules::Input::new(shape).into()
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

                        modules::ConvBn2DInit {
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
                                let config::BatchNorm {
                                    affine,
                                    var_min,
                                    var_max,
                                    ..
                                } = *bn;
                                let mut config = modules::DarkBatchNormInit {
                                    var_min: var_min.map(R64::raw),
                                    var_max: var_max.map(R64::raw),
                                    ..Default::default()
                                };
                                if !affine {
                                    config.ws_init = None;
                                    config.bs_init = None;
                                }
                                config
                            }),
                        }
                        .build(module_path)
                        .into()
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

                        modules::DeconvBn2DInit {
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
                                let config::BatchNorm {
                                    affine,
                                    var_min,
                                    var_max,
                                    ..
                                } = *bn;
                                let mut config = modules::DarkBatchNormInit {
                                    var_min: var_min.map(R64::raw),
                                    var_max: var_max.map(R64::raw),
                                    ..Default::default()
                                };
                                if !affine {
                                    config.ws_init = None;
                                    config.bs_init = None;
                                }
                                config
                            }),
                        }
                        .build(module_path)
                        .into()
                    }
                    config::Module::UpSample2D(config::UpSample2D { ref config, .. }) => {
                        match config {
                            config::UpSample2DConfig::ByScale { scale } => {
                                modules::UpSample2D::new(scale.raw())?.into()
                            }
                            config::UpSample2DConfig::ByStride { .. } => {
                                todo!();
                            }
                        }
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

                        modules::DarkCsp2DInit {
                            in_c,
                            out_c: c,
                            repeat,
                            shortcut,
                            c_mul,
                            batch_norm: bn.enabled.then(|| {
                                let mut config = modules::DarkBatchNormInit::default();
                                if !bn.affine {
                                    config.ws_init = None;
                                    config.bs_init = None;
                                }
                                config
                            }),
                        }
                        .build(module_path)
                        .into()
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

                        modules::SppCsp2DInit {
                            in_c,
                            out_c: c,
                            k: k.to_owned(),
                            c_mul,
                            batch_norm: bn.enabled.then(|| {
                                let mut config = modules::DarkBatchNormInit::default();
                                if !bn.affine {
                                    config.ws_init = None;
                                    config.bs_init = None;
                                }
                                config
                            }),
                        }
                        .build(module_path)
                        .into()
                    }
                    config::Module::Sum2D(_) => modules::Sum2D::new().into(),
                    config::Module::Concat2D(_) => modules::Concat2D::new().into(),
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
                                    RatioSize::from_hw(h.try_into()?, w.try_into()?).unwrap();
                                Ok(size)
                            })
                            .try_collect()?;

                        modules::Detect2DInit {
                            num_classes: classes,
                            anchors,
                        }
                        .build(module_path)
                        .into()
                    }
                    config::Module::MergeDetect2D(_) => modules::MergeDetect2D::new().into(),
                    config::Module::DynamicPad2D(config::DynamicPad2D {
                        r#type,
                        l,
                        r,
                        t,
                        b,
                        ..
                    }) => {
                        let kind = match r#type {
                            config::PaddingKind::Zero => modules::PaddingKind::Zero,
                            config::PaddingKind::Replication => modules::PaddingKind::Replication,
                            config::PaddingKind::Reflection => modules::PaddingKind::Reflection,
                        };
                        modules::DynamicPad::<2>::new(kind, &[l, r, t, b])?.into()
                    }
                    config::Module::DarknetRoute(_) => {
                        todo!();
                    }
                    config::Module::DarknetShortcut(_) => {
                        todo!();
                    }
                    config::Module::MaxPool(_) => {
                        todo!();
                    }
                    config::Module::Linear(_) => {
                        todo!();
                    }
                    config::Module::GroupRef(_) => unreachable!(),
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
                .filter_map(|(&key, layer)| layer.module.is_merge_detect_2d().then(|| key));
            let output_key = iter
                .next()
                .ok_or_else(|| format_err!("no MergeDetect2D layer found"))?;
            ensure!(
                iter.next() == None,
                "the model has multiple MergeDetect2D layers"
            );
            output_key
        };

        Ok(Self { layers, output_key })
    }

    /// Run forward pass.
    pub fn forward_t(
        &mut self,
        input: &Tensor,
        train: bool,
    ) -> Result<tch_goodies::DenseDetectionTensorList> {
        let Self {
            ref mut layers,
            output_key,
        } = *self;
        let mut module_outputs: HashMap<graph::NodeKey, modules::ModuleOutput> = HashMap::new();
        let mut input = Some(input); // it makes sure the input is consumed at most once

        // run the network
        layers.values_mut().try_for_each(|layer| -> Result<_> {
            let Layer {
                key,
                ref mut module,
                ref input_keys,
            } = *layer;

            let module_input: modules::ModuleInput = match input_keys {
                graph::InputKeys::None => modules::ModuleInput::None,
                graph::InputKeys::PlaceHolder => input.take().unwrap().into(),
                graph::InputKeys::Single(src_key) => (&module_outputs[src_key]).try_into()?,
                graph::InputKeys::Indexed(src_keys) => {
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
        let output = module_outputs
            .remove(&output_key)
            .unwrap()
            .merge_detect_2d()
            .unwrap();

        Ok(output)
    }

    pub fn clamp_running_var(&mut self) {
        self.layers.values_mut().for_each(|layer| {
            layer.module.clamp_running_var();
        });
    }

    pub fn denormalize(&mut self) {
        self.layers.values_mut().for_each(|layer| {
            layer.module.denormalize();
        });
    }

    pub fn layers(&self) -> &IndexMap<graph::NodeKey, Layer> {
        &self.layers
    }
}

#[derive(Debug)]
pub struct Layer {
    pub(crate) key: graph::NodeKey,
    pub(crate) module: modules::Module,
    pub(crate) input_keys: graph::InputKeys,
}
