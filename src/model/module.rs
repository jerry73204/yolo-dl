use super::*;
use crate::common::*;

pub enum YoloModule {
    Single(usize, Box<dyn 'static + Fn(&Tensor, bool) -> Tensor>),
    Multi(
        Vec<usize>,
        Box<dyn 'static + Fn(&[&Tensor], bool) -> Tensor>,
    ),
}

impl YoloModule {
    pub fn single<F>(from_index: usize, f: F) -> Self
    where
        F: 'static + Fn(&Tensor, bool) -> Tensor,
    {
        Self::Single(from_index, Box::new(f))
    }

    pub fn multi<F>(from_indexes: Vec<usize>, f: F) -> Self
    where
        F: 'static + Fn(&[&Tensor], bool) -> Tensor,
    {
        Self::Multi(from_indexes, Box::new(f))
    }

    pub fn forward_t<T>(&self, inputs: &[T], train: bool) -> Tensor
    where
        T: Borrow<Tensor>,
    {
        match self {
            Self::Single(_, module_fn) => {
                debug_assert_eq!(inputs.len(), 1);
                module_fn(inputs[0].borrow(), train)
            }
            Self::Multi(_, module_fn) => {
                let inputs = inputs
                    .iter()
                    .map(|tensor| tensor.borrow())
                    .collect::<Vec<_>>();
                module_fn(&inputs, train)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerInit {
    pub name: Option<String>,
    pub export: bool,
    pub kind: LayerKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum LayerKind {
    Focus {
        from: Option<String>,
        out_c: usize,
        k: usize,
    },
    ConvBlock {
        from: Option<String>,
        out_c: usize,
        k: usize,
        s: usize,
    },
    Bottleneck {
        from: Option<String>,
        repeat: usize,
    },
    BottleneckCsp {
        from: Option<String>,
        repeat: usize,
        shortcut: bool,
    },
    Spp {
        from: Option<String>,
        out_c: usize,
        ks: Vec<usize>,
    },
    HeadConv2d {
        from: Option<String>,
        k: usize,
        s: usize,
    },
    Upsample {
        from: Option<String>,
        scale_factor: R64,
    },
    Concat {
        from: Vec<String>,
    },
}

impl LayerKind {
    pub fn from_name(&self) -> Option<&str> {
        match self {
            Self::Focus { from, .. } => from.as_ref().map(|name| name.as_str()),
            Self::ConvBlock { from, .. } => from.as_ref().map(|name| name.as_str()),
            Self::Bottleneck { from, .. } => from.as_ref().map(|name| name.as_str()),
            Self::BottleneckCsp { from, .. } => from.as_ref().map(|name| name.as_str()),
            Self::Spp { from, .. } => from.as_ref().map(|name| name.as_str()),
            Self::HeadConv2d { from, .. } => from.as_ref().map(|name| name.as_str()),
            Self::Upsample { from, .. } => from.as_ref().map(|name| name.as_str()),
            _ => None,
        }
    }

    pub fn from_multiple_names(&self) -> Option<&[String]> {
        let names = match self {
            Self::Concat { from, .. } => from.as_slice(),
            _ => return None,
        };
        Some(names)
    }
}

#[derive(Debug, Clone)]
pub struct ConvBlockInit {
    pub in_c: usize,
    pub out_c: usize,
    pub k: usize,
    pub s: usize,
    pub g: usize,
    pub with_activation: bool,
}

impl ConvBlockInit {
    pub fn new(in_c: usize, out_c: usize) -> Self {
        Self {
            in_c,
            out_c,
            k: 1,
            s: 1,
            g: 1,
            with_activation: true,
        }
    }

    pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor, bool) -> Tensor>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();

        let Self {
            in_c,
            out_c,
            k,
            s,
            g,
            with_activation,
        } = self;

        let conv = nn::conv2d(
            path,
            in_c as i64,
            out_c as i64,
            k as i64,
            nn::ConvConfig {
                stride: s as i64,
                padding: k as i64 / 2,
                groups: g as i64,
                bias: false,
                ..Default::default()
            },
        );
        let bn = nn::batch_norm2d(path, out_c as i64, Default::default());

        Box::new(move |xs, train| {
            let xs = xs.apply(&conv).apply_t(&bn, train);
            if with_activation {
                xs.leaky_relu()
            } else {
                xs
            }
        })
    }
}

#[derive(Debug, Clone)]
pub struct BottleneckInit {
    pub in_c: usize,
    pub out_c: usize,
    pub shortcut: bool,
    pub g: usize,
    pub expansion: R64,
}

impl BottleneckInit {
    pub fn new(in_c: usize, out_c: usize) -> Self {
        Self {
            in_c,
            out_c,
            shortcut: true,
            g: 1,
            expansion: R64::new(0.5),
        }
    }

    pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor, bool) -> Tensor>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();

        let Self {
            in_c,
            out_c,
            shortcut,
            g,
            expansion,
        } = self;

        let intermediate_channels = (out_c as f64 * expansion.raw()) as usize;

        let conv1 = ConvBlockInit {
            k: 1,
            s: 1,
            ..ConvBlockInit::new(in_c, intermediate_channels)
        }
        .build(path);
        let conv2 = ConvBlockInit {
            k: 3,
            s: 1,
            g,
            ..ConvBlockInit::new(intermediate_channels, out_c)
        }
        .build(path);
        let with_add = shortcut && in_c == out_c;

        Box::new(move |xs, train| {
            let ys = conv1(xs, train);
            let ys = conv2(&ys, train);
            if with_add {
                xs + &ys
            } else {
                ys
            }
        })
    }
}

#[derive(Debug, Clone)]
pub struct BottleneckCspInit {
    pub in_c: usize,
    pub out_c: usize,
    pub repeat: usize,
    pub shortcut: bool,
    pub g: usize,
    pub expansion: R64,
}

impl BottleneckCspInit {
    pub fn new(in_c: usize, out_c: usize) -> Self {
        Self {
            in_c,
            out_c,
            repeat: 1,
            shortcut: true,
            g: 1,
            expansion: R64::new(0.5),
        }
    }

    pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor, bool) -> Tensor>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();

        let Self {
            in_c,
            out_c,
            repeat,
            shortcut,
            g,
            expansion,
        } = self;
        debug_assert!(repeat > 0);

        let intermediate_channels = (out_c as f64 * expansion.raw()) as usize;

        let conv1 = ConvBlockInit {
            k: 1,
            s: 1,
            ..ConvBlockInit::new(in_c, intermediate_channels)
        }
        .build(path);
        let conv2 = nn::conv2d(
            path,
            in_c as i64,
            intermediate_channels as i64,
            1,
            nn::ConvConfig {
                stride: 1,
                bias: false,
                ..Default::default()
            },
        );
        let conv3 = nn::conv2d(
            path,
            intermediate_channels as i64,
            intermediate_channels as i64,
            1,
            nn::ConvConfig {
                stride: 1,
                bias: false,
                ..Default::default()
            },
        );
        let conv4 = ConvBlockInit {
            k: 1,
            s: 1,
            ..ConvBlockInit::new(out_c, out_c)
        }
        .build(path);
        let bn = nn::batch_norm2d(path, intermediate_channels as i64 * 2, Default::default());
        let bottlenecks = (0..repeat)
            .map(|_| {
                BottleneckInit {
                    shortcut,
                    g,
                    expansion: R64::new(1.0),
                    ..BottleneckInit::new(intermediate_channels, intermediate_channels)
                }
                .build(path)
            })
            .collect::<Vec<_>>();

        Box::new(move |xs, train| {
            let y1 = {
                let y = conv1(xs, train);
                let y = bottlenecks
                    .iter()
                    .fold(y, |input, block| block(&input, train));
                y.apply_t(&conv3, train)
            };
            let y2 = xs.apply_t(&conv2, train);
            conv4(
                &Tensor::cat(&[y1, y2], 1).apply_t(&bn, train).leaky_relu(),
                train,
            )
        })
    }
}

pub struct SppInit {
    pub in_c: usize,
    pub out_c: usize,
    pub ks: Vec<usize>,
}

impl SppInit {
    pub fn new(in_c: usize, out_c: usize) -> Self {
        Self {
            in_c,
            out_c,
            ks: vec![5, 9, 13],
        }
    }

    pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor, bool) -> Tensor>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();

        let Self { in_c, out_c, ks } = self;
        let intermediate_channels = in_c / 2;

        let conv1 = ConvBlockInit {
            k: 1,
            s: 1,
            ..ConvBlockInit::new(in_c, intermediate_channels)
        }
        .build(path);

        let conv2 = ConvBlockInit {
            k: 1,
            s: 1,
            ..ConvBlockInit::new(intermediate_channels * (ks.len() + 1), out_c)
        }
        .build(path);

        Box::new(move |xs, train| {
            let transformed_xs = conv1(xs, train);

            let pyramid_iter = ks.iter().cloned().map(|k| {
                let k = k as i64;
                let padding = k / 2;
                let s = 1;
                let dilation = 1;
                let ceil_mode = false;
                transformed_xs.max_pool2d(
                    &[k, k],
                    &[s, s],
                    &[padding, padding],
                    &[dilation, dilation],
                    ceil_mode,
                )
            });
            let cat_xs = Tensor::cat(
                &iter::once(transformed_xs.shallow_clone())
                    .chain(pyramid_iter)
                    .collect::<Vec<_>>(),
                1,
            );

            conv2(&cat_xs, train)
        })
    }
}

#[derive(Debug, Clone)]
pub struct FocusInit {
    pub in_c: usize,
    pub out_c: usize,
    pub k: usize,
}

impl FocusInit {
    pub fn new(in_c: usize, out_c: usize) -> Self {
        Self { in_c, out_c, k: 1 }
    }

    pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor, bool) -> Tensor>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();

        let Self { in_c, out_c, k } = self;

        let conv = ConvBlockInit {
            k,
            s: 1,
            ..ConvBlockInit::new(in_c * 4, out_c)
        }
        .build(path);

        Box::new(move |xs, train| {
            let (height, width) = match xs.size().as_slice() {
                &[_bsize, _channels, height, width] => (height, width),
                _ => unreachable!(),
            };

            let xs = Tensor::cat(
                &[
                    xs.slice(2, 0, height, 2).slice(3, 0, width, 2),
                    xs.slice(2, 1, height, 2).slice(3, 0, width, 2),
                    xs.slice(2, 0, height, 2).slice(3, 1, width, 2),
                    xs.slice(2, 1, height, 2).slice(3, 1, width, 2),
                ],
                1,
            );
            conv(&xs, train)
        })
    }
}

#[derive(Debug, Clone)]
pub struct DetectInit {
    pub num_classes: usize,
    pub anchors: Vec<Vec<(usize, usize)>>,
}

impl DetectInit {
    pub fn build<'p, P>(self, path: P) -> Box<dyn FnMut(&[&Tensor], bool, i64, i64) -> YoloOutput>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();
        let device = path.device();

        let Self {
            num_classes,
            anchors,
        } = self;

        assert!(anchors
            .iter()
            .all(|anchor| anchor.len() == anchors[0].len()));
        let num_anchors = anchors[0].len() as i64;
        let num_outputs_per_anchor = num_classes as i64 + 5;
        let num_detections = anchors.len() as i64;

        let anchor_size_multipliers = anchors
            .iter()
            .cloned()
            .map(|sizes| {
                Tensor::of_slice(
                    &sizes
                        .iter()
                        .cloned()
                        .flat_map(|(y, x)| vec![y, x])
                        .map(|component| component as i64)
                        .collect::<Vec<_>>(),
                )
                .to_device(device)
                .view([1, num_anchors, 1, 1, 2])
            })
            .collect::<Vec<_>>();

        Box::new(
            move |tensors: &[&Tensor], train: bool, image_height: i64, image_width: i64| {
                debug_assert_eq!(tensors.len() as i64, num_detections);

                let feature_maps = tensors
                    .iter()
                    .cloned()
                    .map(|xs| {
                        let (batch_size, channels, height, width) = xs.size4().unwrap();
                        debug_assert_eq!(channels, num_anchors * num_outputs_per_anchor);

                        let outputs = xs
                            .view(&[
                                batch_size,
                                num_anchors,
                                num_outputs_per_anchor,
                                height,
                                width,
                            ] as &[_])
                            .permute(&[0, 1, 3, 4, 2]);

                        // output shape [bsize, n_anchors, height, width, n_outputs]
                        outputs
                    })
                    .collect::<Vec<_>>();

                YoloOutput::new(
                    image_height,
                    image_width,
                    anchor_size_multipliers.shallow_clone(),
                    feature_maps,
                )
            },
        )
    }
}