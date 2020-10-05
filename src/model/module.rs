use super::*;
use crate::{
    common::*,
    utils::{GridSize, PixelSize, Unzip7},
};

pub enum YoloModule {
    Single(usize, Box<dyn 'static + Fn(&Tensor, bool) -> Tensor + Send>),
    Multi(
        Vec<usize>,
        Box<dyn 'static + Fn(&[&Tensor], bool) -> Tensor + Send>,
    ),
}

impl fmt::Debug for YoloModule {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            Self::Single(from_index, _) => f
                .debug_tuple("Single")
                .field(&from_index)
                .field(&"func")
                .finish(),
            Self::Multi(from_indexes, _) => f
                .debug_tuple("Multi")
                .field(&from_indexes)
                .field(&"func")
                .finish(),
        }
    }
}

impl YoloModule {
    pub fn single<F>(from_index: usize, f: F) -> Self
    where
        F: 'static + Fn(&Tensor, bool) -> Tensor + Send,
    {
        Self::Single(from_index, Box::new(f))
    }

    pub fn multi<F>(from_indexes: Vec<usize>, f: F) -> Self
    where
        F: 'static + Fn(&[&Tensor], bool) -> Tensor + Send,
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

    pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor, bool) -> Tensor + Send>
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

    pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor, bool) -> Tensor + Send>
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

    pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor, bool) -> Tensor + Send>
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

    pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor, bool) -> Tensor + Send>
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

    pub fn build<'p, P>(self, path: P) -> Box<dyn Fn(&Tensor, bool) -> Tensor + Send>
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
    pub anchors_list: Vec<Vec<PixelSize<usize>>>,
}

impl DetectInit {
    pub fn build<'p, P>(self, path: P) -> DetectModule
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();
        let device = path.device();

        let Self {
            num_classes,
            anchors_list,
        } = self;

        let anchors_list: Vec<Vec<_>> = anchors_list
            .into_iter()
            .map(|list| {
                list.into_iter()
                    .map(|PixelSize { height, width, .. }| {
                        PixelSize::new(height as i64, width as i64)
                    })
                    .collect()
            })
            .collect();

        DetectModule {
            num_classes: num_classes as i64,
            anchors_list: Arc::new(anchors_list),
            device,
            cache: HashMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct DetectModule {
    num_classes: i64,
    anchors_list: Arc<Vec<Vec<PixelSize<i64>>>>,
    device: Device,
    cache: HashMap<PixelSize<i64>, Arc<DetectModuleCache>>,
}

impl DetectModule {
    pub fn forward_t(
        &mut self,
        tensors: &[&Tensor],
        _train: bool,
        image_size: &PixelSize<i64>,
    ) -> YoloOutput {
        debug_assert_eq!(tensors.len(), self.anchors_list.len());
        let num_outputs_per_anchor = self.num_classes + 5;
        let (batch_size, _channels, _height, _width) = tensors[0].size4().unwrap();

        // load cached data
        let DetectModuleCache {
            positions_grids,
            anchor_sizes_list,
            anchor_sizes_grids,
        } = &*self.get_cache(tensors, image_size);

        // compute outputs
        let (cy_vec, cx_vec, h_vec, w_vec, objectness_vec, classification_vec, layer_meta) = izip!(
            tensors.iter(),
            positions_grids,
            anchor_sizes_list,
            anchor_sizes_grids
        )
        .scan(
            0,
            |base_flat_index, (xs, positions_grid, anchor_sizes, anchor_sizes_grid)| {
                let (bsize, channels, feature_height, feature_width) = xs.size4().unwrap();
                let num_anchors = anchor_sizes.len() as i64;
                debug_assert_eq!(bsize, batch_size);
                debug_assert_eq!(channels, num_anchors * num_outputs_per_anchor);
                let feature_size = GridSize::new(feature_height, feature_width);

                // gride size in pixels
                let grid_height = image_size.height as f64 / feature_height as f64;
                let grid_width = image_size.width as f64 / feature_width as f64;

                let layer_meta = {
                    let begin_flat_index = *base_flat_index;
                    *base_flat_index += num_anchors * feature_height * feature_width;
                    let end_flat_index = *base_flat_index;

                    // compute base flat index
                    let layer_meta = LayerMeta {
                        feature_size: feature_size.to_owned(),
                        grid_size: PixelSize::new(grid_height, grid_width),
                        anchors: anchor_sizes.to_owned(),
                        begin_flat_index,
                        end_flat_index,
                    };
                    layer_meta
                };

                // transform outputs
                let (cy, cx, h, w, objectness, classification) = {
                    // convert shape to [n_anchors, height, width, batch_size, n_outputs]
                    let outputs = xs
                        .view([
                            batch_size,
                            num_anchors,
                            num_outputs_per_anchor,
                            feature_height,
                            feature_width,
                        ])
                        .permute(&[1, 3, 4, 0, 2]);

                    let sigmoid = outputs.sigmoid();

                    // positions in grid units
                    let position = sigmoid.i((.., .., .., .., 0..2)) * 2.0 - 0.5 + positions_grid;
                    let cy = position.i((.., .., .., .., 0..1));
                    let cx = position.i((.., .., .., .., 1..2));

                    // bbox sizes in grid units
                    let size = sigmoid.i((.., .., .., .., 2..4)).pow(2.0) * anchor_sizes_grid;
                    let h = size.i((.., .., .., .., 0..1));
                    let w = size.i((.., .., .., .., 1..2));

                    // objectness
                    let objectness = sigmoid.i((.., .., .., .., 4..5));

                    // sparse classification
                    let classification = sigmoid.i((.., .., .., .., 5..));

                    let cy = cy.view([-1, batch_size, 1]);
                    let cx = cx.view([-1, batch_size, 1]);
                    let h = h.view([-1, batch_size, 1]);
                    let w = w.view([-1, batch_size, 1]);
                    let objectness = objectness.view([-1, batch_size, 1]);
                    let classification = classification.view([-1, batch_size, self.num_classes]);

                    (cy, cx, h, w, objectness, classification)
                };

                Some((cy, cx, h, w, objectness, classification, layer_meta))
            },
        )
        .unzip_n_vec();

        let cy = Tensor::cat(&cy_vec, 0).view([-1, 1]);
        let cx = Tensor::cat(&cx_vec, 0).view([-1, 1]);
        let h = Tensor::cat(&h_vec, 0).view([-1, 1]);
        let w = Tensor::cat(&w_vec, 0).view([-1, 1]);
        let objectness = Tensor::cat(&objectness_vec, 0).view([-1, 1]);
        let classification = Tensor::cat(&classification_vec, 0).view([-1, self.num_classes]);

        YoloOutput {
            image_size: image_size.to_owned(),
            batch_size,
            num_classes: self.num_classes,
            device: self.device,
            cy,
            cx,
            height: h,
            width: w,
            objectness,
            classification,
            layer_meta,
        }
    }

    fn get_cache(
        &mut self,
        tensors: &[&Tensor],
        image_size: &PixelSize<i64>,
    ) -> Arc<DetectModuleCache> {
        let device = self.device;
        let anchors_list = self.anchors_list.clone();

        self.cache
            .entry(image_size.to_owned())
            .or_insert_with(|| {
                let positions_grids = {
                    tensors
                        .iter()
                        .map(|xs| {
                            let (_bsize, _channels, feature_height, feature_width) =
                                xs.size4().unwrap();
                            tch::no_grad(|| {
                                let grid = {
                                    let grids = Tensor::meshgrid(&[
                                        Tensor::arange(feature_height, (Kind::Float, device)),
                                        Tensor::arange(feature_width, (Kind::Float, device)),
                                    ]);
                                    Tensor::stack(&[&grids[0], &grids[1]], 2).view(&[
                                        1,
                                        feature_height,
                                        feature_width,
                                        1,
                                        2,
                                    ]
                                        as &[_])
                                };
                                grid.set_requires_grad(false)
                            })
                        })
                        .collect_vec()
                };

                let anchor_sizes_list = {
                    anchors_list
                        .iter()
                        .zip_eq(tensors.iter().cloned())
                        .map(|(anchors, xs)| {
                            let (_bsize, _channels, feature_height, feature_width) =
                                xs.size4().unwrap();

                            // gride size in pixels
                            let grid_height = image_size.height as f64 / feature_height as f64;
                            let grid_width = image_size.width as f64 / feature_width as f64;

                            anchors
                                .iter()
                                .cloned()
                                .map(|pixel_size| {
                                    let PixelSize {
                                        height: pixel_height,
                                        width: pixel_width,
                                        ..
                                    } = pixel_size;
                                    GridSize::new(
                                        pixel_height as f64 / grid_height,
                                        pixel_width as f64 / grid_width,
                                    )
                                })
                                .collect_vec()
                        })
                        .collect_vec()
                };

                let anchor_sizes_grids = {
                    anchor_sizes_list
                        .iter()
                        .map(|anchor_sizes| {
                            let num_anchors = anchor_sizes.len();
                            let components = anchor_sizes
                                .iter()
                                .cloned()
                                .flat_map(|anchor_size| {
                                    let GridSize {
                                        height: anchor_height,
                                        width: anchor_width,
                                        ..
                                    } = anchor_size;
                                    vec![anchor_height as f32, anchor_width as f32]
                                })
                                .collect_vec();

                            Tensor::of_slice(&components)
                                .to_device(device)
                                .view([num_anchors as i64, 1, 1, 1, 2])
                                .set_requires_grad(false)
                        })
                        .collect_vec()
                };

                Arc::new(DetectModuleCache {
                    positions_grids,
                    anchor_sizes_list,
                    anchor_sizes_grids,
                })
            })
            .clone()
    }
}

#[derive(Debug)]
struct DetectModuleCache {
    positions_grids: Vec<Tensor>,
    anchor_sizes_list: Vec<Vec<GridSize<f64>>>,
    anchor_sizes_grids: Vec<Tensor>,
}
