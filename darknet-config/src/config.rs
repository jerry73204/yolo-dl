use crate::common::*;

pub use batch_norm_config::*;
pub use connected_config::*;
pub use convolutional_config::*;
pub use darknet_config::*;
pub use dropout_config::*;
pub use gaussian_yolo_config::*;
pub use items::*;
pub use max_pool_config::*;
pub use misc::*;
pub use net_config::*;
pub use route_config::*;
pub use shortcut_config::*;
pub use softmax_config::*;
pub use up_sample_config::*;
pub use yolo_config::*;

mod darknet_config {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(try_from = "Items")]
    pub struct DarknetConfig {
        pub net: NetConfig,
        pub layers: Vec<LayerConfig>,
    }

    impl DarknetConfig {
        pub fn load<P>(config_file: P) -> Result<Self>
        where
            P: AsRef<Path>,
        {
            Ok(Self::from_str(&fs::read_to_string(config_file)?)?)
        }

        pub fn to_string(&self) -> Result<String> {
            Ok(serde_ini::to_string(self)?)
        }
    }

    impl FromStr for DarknetConfig {
        type Err = Error;

        fn from_str(text: &str) -> Result<Self, Self::Err> {
            Ok(serde_ini::from_str(text)?)
        }
    }

    impl TryFrom<Items> for DarknetConfig {
        type Error = anyhow::Error;

        fn try_from(Items(items): Items) -> Result<Self, Self::Error> {
            // ensure only the first item is "net" item
            {
                let mut iter = items.iter();
                ensure!(
                    matches!(iter.next(), Some(Item::Net(_))),
                    "the first item must be [net]"
                );
                ensure!(
                    iter.all(|item| !matches!(item, Item::Net(_))),
                    "net item must be the first item"
                );
            };

            // extract global options from yolo item
            let classes = {
                let (classes_vec, anchors_vec) = items
                    .iter()
                    .filter_map(|item| match item {
                        Item::Yolo(yolo) => {
                            let RawYoloConfig {
                                classes,
                                ref anchors,
                                ..
                            } = *yolo;

                            Some((classes, anchors))
                        }
                        Item::GaussianYolo(yolo) => {
                            let RawGaussianYoloConfig {
                                classes,
                                ref anchors,
                                ..
                            } = *yolo;

                            Some((classes, anchors))
                        }
                        _ => None,
                    })
                    .unzip_n_vec();

                let classes = {
                    let classes_set: HashSet<_> = classes_vec.iter().cloned().collect();
                    ensure!(
                        classes_set.len() == 1,
                        "the classes of every yolo layer must be equal"
                    );
                    classes_vec[0]
                };

                {
                    let anchors_set: HashSet<_> = anchors_vec.iter().collect();
                    ensure!(
                        anchors_set.len() == 1,
                        "the anchors of every yolo layer must be equal"
                    );
                }

                classes
            };

            let mut items_iter = items.into_iter();

            // build net item
            let net = {
                let net = match items_iter.next().unwrap() {
                    Item::Net(net) => net,
                    _ => unreachable!(),
                };

                let RawNetConfig {
                    max_batches,
                    batch,
                    learning_rate,
                    learning_rate_min,
                    sgdr_cycle,
                    sgdr_mult,
                    momentum,
                    decay,
                    subdivisions,
                    time_steps,
                    track,
                    augment_speed,
                    sequential_subdivisions,
                    try_fix_nan,
                    loss_scale,
                    dynamic_minibatch,
                    optimized_memory,
                    workspace_size_limit_mb,
                    adam,
                    b1,
                    b2,
                    eps,
                    width,
                    height,
                    channels,
                    inputs,
                    max_crop,
                    min_crop,
                    flip,
                    blur,
                    gaussian_noise,
                    mixup,
                    cutmux,
                    mosaic,
                    letter_box,
                    mosaic_bound,
                    contrastive,
                    contrastive_jit_flip,
                    contrastive_color,
                    unsupervised,
                    label_smooth_eps,
                    resize_step,
                    attention,
                    adversarial_lr,
                    max_chart_loss,
                    angle,
                    aspect,
                    saturation,
                    exposure,
                    hue,
                    power,
                    policy,
                    burn_in,
                    step,
                    scale,
                    steps,
                    scales,
                    seq_scales,
                    gamma,
                } = net;

                let sgdr_cycle = sgdr_cycle.unwrap_or(max_batches);
                let sequential_subdivisions = sequential_subdivisions.unwrap_or(subdivisions);
                let adam = if adam {
                    Some(Adam { b1, b2, eps })
                } else {
                    None
                };
                let max_crop = max_crop.unwrap_or_else(|| width.map(|w| w.get()).unwrap_or(0) * 2);
                let min_crop = min_crop.unwrap_or_else(|| width.map(|w| w.get()).unwrap_or(0));
                let input_size = match (inputs, height, width, channels) {
                    (Some(inputs), None, None, None) => Shape::Flat(inputs.get()),
                    (None, Some(height), Some(width), Some(channels)) => {
                        Shape::Hwc([height.get(), width.get(), channels.get()])
                    }
                    _ => bail!("either inputs or height/width/channels must be specified"),
                };
                let policy = match policy {
                    PolicyKind::Random => Policy::Random,
                    PolicyKind::Poly => Policy::Poly,
                    PolicyKind::Constant => Policy::Constant,
                    PolicyKind::Step => Policy::Step { step, scale },
                    PolicyKind::Exp => Policy::Exp { gamma },
                    PolicyKind::Sigmoid => Policy::Sigmoid { gamma, step },
                    PolicyKind::Steps => {
                        let steps = steps.ok_or_else(|| {
                            format_err!("steps must be specified for step policy")
                        })?;
                        let scales = {
                            let scales = scales.ok_or_else(|| {
                                format_err!("scales must be specified for step policy")
                            })?;
                            ensure!(
                                steps.len() == scales.len(),
                                "the length of steps and scales must be equal"
                            );
                            scales
                        };
                        let seq_scales = {
                            let seq_scales =
                                seq_scales.unwrap_or_else(|| vec![R64::new(1.0); steps.len()]);
                            ensure!(
                                steps.len() == seq_scales.len(),
                                "the length of steps and seq_scales must be equal"
                            );
                            seq_scales
                        };

                        Policy::Steps {
                            steps,
                            scales,
                            seq_scales,
                        }
                    }
                    PolicyKind::Sgdr => match (steps, scales, seq_scales) {
                        (Some(steps), scales, seq_scales) => {
                            let scales = scales.unwrap_or_else(|| vec![R64::new(1.0); steps.len()]);
                            let seq_scales =
                                seq_scales.unwrap_or_else(|| vec![R64::new(1.0); steps.len()]);
                            ensure!(
                                steps.len() == scales.len(),
                                "the length of steps and scales must be equal"
                            );
                            ensure!(
                                steps.len() == seq_scales.len(),
                                "the length of steps and seq_scales must be equal"
                            );

                            Policy::SgdrCustom {
                                steps,
                                scales,
                                seq_scales,
                            }
                        }
                        (None, None, None) => Policy::Sgdr,
                        _ => bail!("either none or at least steps must be specifeid"),
                    },
                };

                NetConfig {
                    max_batches,
                    batch,
                    learning_rate,
                    learning_rate_min,
                    sgdr_cycle,
                    sgdr_mult,
                    momentum,
                    decay,
                    subdivisions,
                    time_steps,
                    track,
                    augment_speed,
                    sequential_subdivisions,
                    try_fix_nan,
                    loss_scale,
                    dynamic_minibatch,
                    optimized_memory,
                    workspace_size_limit_mb,
                    adam,
                    input_size,
                    max_crop,
                    min_crop,
                    flip,
                    blur,
                    gaussian_noise,
                    mixup,
                    cutmux,
                    mosaic,
                    letter_box,
                    mosaic_bound,
                    contrastive,
                    contrastive_jit_flip,
                    contrastive_color,
                    unsupervised,
                    label_smooth_eps,
                    resize_step,
                    attention,
                    adversarial_lr,
                    max_chart_loss,
                    angle,
                    aspect,
                    saturation,
                    exposure,
                    hue,
                    power,
                    policy,
                    burn_in,
                    classes,
                }
            };

            // build layers
            let layers: Vec<_> = items_iter
                .map(|item| {
                    let layer = match item {
                        Item::Connected(layer) => LayerConfig::Connected(layer),
                        Item::Convolutional(layer) => LayerConfig::Convolutional(layer),
                        Item::Route(layer) => LayerConfig::Route(layer),
                        Item::Shortcut(layer) => LayerConfig::Shortcut(layer),
                        Item::MaxPool(layer) => LayerConfig::MaxPool(layer),
                        Item::UpSample(layer) => LayerConfig::UpSample(layer),
                        Item::BatchNorm(layer) => LayerConfig::BatchNorm(layer),
                        Item::Dropout(layer) => LayerConfig::Dropout(layer),
                        Item::Softmax(layer) => LayerConfig::Softmax(layer.try_into()?),
                        Item::GaussianYolo(layer) => {
                            let RawGaussianYoloConfig {
                                classes,
                                num,
                                mask,
                                max_boxes,
                                max_delta,
                                counters_per_class,
                                label_smooth_eps,
                                scale_x_y,
                                objectness_smooth,
                                uc_normalizer,
                                iou_normalizer,
                                obj_normalizer,
                                cls_normalizer,
                                delta_normalizer,
                                iou_loss,
                                iou_thresh_kind,
                                beta_nms,
                                nms_kind,
                                yolo_point,
                                jitter,
                                resize,
                                ignore_thresh,
                                truth_thresh,
                                iou_thresh,
                                random,
                                map,
                                anchors,
                                common,
                            } = layer;

                            let mask = mask.unwrap_or_else(|| IndexSet::new());
                            let anchors = match (num, anchors) {
                                (0, None) => vec![],
                                (_, None) => bail!("num and length of anchors mismatch"),
                                (_, Some(anchors)) => {
                                    ensure!(
                                        anchors.len() == num as usize,
                                        "num and length of anchors mismatch"
                                    );
                                    let anchors: Vec<_> = mask
                                        .into_iter()
                                        .map(|index| -> Result<_> {
                                            Ok(anchors.get(index as usize).ok_or_else(|| format_err!("mask index exceeds total number of anchors"))?.clone())
                                        })
                                        .try_collect()?;
                                    anchors
                                }
                            };

                            LayerConfig::GaussianYolo(GaussianYoloConfig {
                                max_boxes,
                                max_delta,
                                counters_per_class,
                                label_smooth_eps,
                                scale_x_y,
                                objectness_smooth,
                                uc_normalizer,
                                iou_normalizer,
                                obj_normalizer,
                                cls_normalizer,
                                delta_normalizer,
                                iou_loss,
                                iou_thresh_kind,
                                beta_nms,
                                nms_kind,
                                yolo_point,
                                jitter,
                                resize,
                                ignore_thresh,
                                truth_thresh,
                                iou_thresh,
                                random,
                                map,
                                anchors,
                                common,
                            })
                        }
                        Item::Yolo(layer) => {
                            let RawYoloConfig {
                                classes,
                                num,
                                mask,
                                max_boxes,
                                max_delta,
                                counters_per_class,
                                label_smooth_eps,
                                scale_x_y,
                                objectness_smooth,
                                iou_normalizer,
                                obj_normalizer,
                                cls_normalizer,
                                delta_normalizer,
                                iou_loss,
                                iou_thresh_kind,
                                beta_nms,
                                nms_kind,
                                yolo_point,
                                jitter,
                                resize,
                                focal_loss,
                                ignore_thresh,
                                truth_thresh,
                                iou_thresh,
                                random,
                                track_history_size,
                                sim_thresh,
                                dets_for_track,
                                dets_for_show,
                                track_ciou_norm,
                                embedding_layer,
                                map,
                                anchors,
                                common,
                            } = layer;

                            let mask = mask.unwrap_or_else(|| IndexSet::new());
                            let anchors = match (num, anchors) {
                                (0, None) => vec![],
                                (_, None) => bail!("num and length of anchors mismatch"),
                                (_, Some(anchors)) => {
                                    ensure!(
                                        anchors.len() == num as usize,
                                        "num and length of anchors mismatch"
                                    );
                                    let anchors: Vec<_> = mask
                                        .into_iter()
                                        .map(|index| -> Result<_> {
                                            Ok(anchors.get(index as usize).ok_or_else(|| format_err!("mask index exceeds total number of anchors"))?.clone())
                                        })
                                        .try_collect()?;
                                    anchors
                                }
                            };

                            LayerConfig::Yolo(YoloConfig {
                                max_boxes,
                                max_delta,
                                counters_per_class,
                                label_smooth_eps,
                                scale_x_y,
                                objectness_smooth,
                                iou_normalizer,
                                obj_normalizer,
                                cls_normalizer,
                                delta_normalizer,
                                iou_loss,
                                iou_thresh_kind,
                                beta_nms,
                                nms_kind,
                                yolo_point,
                                jitter,
                                resize,
                                focal_loss,
                                ignore_thresh,
                                truth_thresh,
                                iou_thresh,
                                random,
                                track_history_size,
                                sim_thresh,
                                dets_for_track,
                                dets_for_show,
                                track_ciou_norm,
                                embedding_layer,
                                map,
                                anchors,
                                common,
                            })
                        }
                        Item::Net(_layer) => {
                            bail!("the 'net' layer must appear in the first section")
                        }
                    };
                    Ok(layer)
                })
                .try_collect()?;

            Ok(Self { net, layers })
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum LayerConfig {
        #[serde(rename = "connected")]
        Connected(ConnectedConfig),
        #[serde(rename = "convolutional")]
        Convolutional(ConvolutionalConfig),
        #[serde(rename = "route")]
        Route(RouteConfig),
        #[serde(rename = "shortcut")]
        Shortcut(ShortcutConfig),
        #[serde(rename = "maxpool")]
        MaxPool(MaxPoolConfig),
        #[serde(rename = "upsample")]
        UpSample(UpSampleConfig),
        #[serde(rename = "batchnorm")]
        BatchNorm(BatchNormConfig),
        #[serde(rename = "dropout")]
        Dropout(DropoutConfig),
        #[serde(rename = "dropout")]
        Softmax(SoftmaxConfig),
        #[serde(rename = "yolo")]
        Yolo(YoloConfig),
        #[serde(rename = "Gaussian_yolo")]
        GaussianYolo(GaussianYoloConfig),
    }

    impl LayerConfig {
        pub fn common(&self) -> &CommonLayerOptions {
            match self {
                LayerConfig::Connected(layer) => &layer.common,
                LayerConfig::Convolutional(layer) => &layer.common,
                LayerConfig::Route(layer) => &layer.common,
                LayerConfig::Shortcut(layer) => &layer.common,
                LayerConfig::MaxPool(layer) => &layer.common,
                LayerConfig::UpSample(layer) => &layer.common,
                LayerConfig::BatchNorm(layer) => &layer.common,
                LayerConfig::Dropout(layer) => &layer.common,
                LayerConfig::Softmax(layer) => &layer.common,
                LayerConfig::Yolo(layer) => &layer.common,
                LayerConfig::GaussianYolo(layer) => &layer.common,
            }
        }
    }
}

mod net_config {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct NetConfig {
        pub max_batches: u64,
        pub batch: u64,
        pub learning_rate: R64,
        pub learning_rate_min: R64,
        pub sgdr_cycle: u64,
        pub sgdr_mult: u64,
        pub momentum: R64,
        pub decay: R64,
        pub subdivisions: u64,
        pub time_steps: u64,
        pub track: u64,
        pub augment_speed: u64,
        pub sequential_subdivisions: u64,
        pub try_fix_nan: bool,
        pub loss_scale: R64,
        pub dynamic_minibatch: bool,
        pub optimized_memory: bool,
        pub workspace_size_limit_mb: u64,
        pub adam: Option<Adam>,
        pub input_size: Shape,
        pub max_crop: u64,
        pub min_crop: u64,
        pub flip: bool,
        pub blur: bool,
        pub gaussian_noise: bool,
        pub mixup: MixUp,
        pub cutmux: bool,
        pub mosaic: bool,
        pub letter_box: bool,
        pub mosaic_bound: bool,
        pub contrastive: bool,
        pub contrastive_jit_flip: bool,
        pub contrastive_color: bool,
        pub unsupervised: bool,
        pub label_smooth_eps: R64,
        pub resize_step: u64,
        pub attention: bool,
        pub adversarial_lr: R64,
        pub max_chart_loss: R64,
        pub angle: R64,
        pub aspect: R64,
        pub saturation: R64,
        pub exposure: R64,
        pub hue: R64,
        pub power: R64,
        pub policy: Policy,
        pub burn_in: u64,
        pub classes: u64,
    }

    impl NetConfig {
        pub fn iteration(&self, seen: u64) -> u64 {
            seen / (self.batch * self.subdivisions)
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct RawNetConfig {
        #[serde(default = "defaults::max_batches")]
        pub max_batches: u64,
        #[serde(default = "defaults::batch")]
        pub batch: u64,
        #[serde(default = "defaults::learning_rate")]
        pub learning_rate: R64,
        #[serde(default = "defaults::learning_rate_min")]
        pub learning_rate_min: R64,
        pub sgdr_cycle: Option<u64>,
        #[serde(default = "defaults::sgdr_mult")]
        pub sgdr_mult: u64,
        #[serde(default = "defaults::momentum")]
        pub momentum: R64,
        #[serde(default = "defaults::decay")]
        pub decay: R64,
        #[serde(default = "defaults::subdivisions")]
        pub subdivisions: u64,
        #[serde(default = "defaults::time_steps")]
        pub time_steps: u64,
        #[serde(default = "defaults::track")]
        pub track: u64,
        #[serde(default = "defaults::augment_speed")]
        pub augment_speed: u64,
        pub sequential_subdivisions: Option<u64>,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub try_fix_nan: bool,
        #[serde(default = "defaults::loss_scale")]
        pub loss_scale: R64,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub dynamic_minibatch: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub optimized_memory: bool,
        #[serde(
            rename = "workspace_size_limit_MB",
            default = "defaults::workspace_size_limit_mb"
        )]
        pub workspace_size_limit_mb: u64,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub adam: bool,
        #[serde(rename = "B1", default = "defaults::b1")]
        pub b1: R64,
        #[serde(rename = "B2", default = "defaults::b2")]
        pub b2: R64,
        #[serde(default = "defaults::eps")]
        pub eps: R64,
        pub width: Option<NonZeroU64>,
        pub height: Option<NonZeroU64>,
        pub channels: Option<NonZeroU64>,
        pub inputs: Option<NonZeroU64>,
        pub max_crop: Option<u64>,
        pub min_crop: Option<u64>,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_true")]
        pub flip: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub blur: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub gaussian_noise: bool,
        #[serde(default = "defaults::mixup")]
        pub mixup: MixUp,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub cutmux: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub mosaic: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub letter_box: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub mosaic_bound: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub contrastive: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub contrastive_jit_flip: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub contrastive_color: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub unsupervised: bool,
        #[serde(default = "defaults::label_smooth_eps")]
        pub label_smooth_eps: R64,
        #[serde(default = "defaults::resize_step")]
        pub resize_step: u64,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub attention: bool,
        #[serde(default = "defaults::adversarial_lr")]
        pub adversarial_lr: R64,
        #[serde(default = "defaults::max_chart_loss")]
        pub max_chart_loss: R64,
        #[serde(default = "defaults::angle")]
        pub angle: R64,
        #[serde(default = "defaults::aspect")]
        pub aspect: R64,
        #[serde(default = "defaults::saturation")]
        pub saturation: R64,
        #[serde(default = "defaults::exposure")]
        pub exposure: R64,
        #[serde(default = "defaults::hue")]
        pub hue: R64,
        #[serde(default = "defaults::power")]
        pub power: R64,
        #[serde(default = "defaults::policy")]
        pub policy: PolicyKind,
        #[serde(default = "defaults::burn_in")]
        pub burn_in: u64,
        #[serde(default = "defaults::step")]
        pub step: u64,
        #[serde(default = "defaults::scale")]
        pub scale: R64,
        #[serde(with = "serde_opt_vec_u64", default)]
        pub steps: Option<Vec<u64>>,
        #[serde(with = "serde_opt_vec_r64", default)]
        pub scales: Option<Vec<R64>>,
        #[serde(with = "serde_opt_vec_r64", default)]
        pub seq_scales: Option<Vec<R64>>,
        #[serde(default = "defaults::gamma")]
        pub gamma: R64,
    }
}

mod connected_config {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct ConnectedConfig {
        #[serde(default = "defaults::connected_output")]
        pub output: u64,
        #[serde(default = "defaults::connected_activation")]
        pub activation: Activation,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub batch_normalize: bool,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }
}

mod convolutional_config {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(try_from = "RawConvolutionalConfig", into = "RawConvolutionalConfig")]
    pub struct ConvolutionalConfig {
        pub filters: u64,
        pub groups: u64,
        pub size: u64,
        pub batch_normalize: bool,
        pub stride_x: u64,
        pub stride_y: u64,
        pub dilation: u64,
        pub antialiasing: bool,
        pub padding: u64,
        pub activation: Activation,
        pub assisted_excitation: bool,
        pub share_index: Option<LayerIndex>,
        pub cbn: bool,
        pub binary: bool,
        pub xnor: bool,
        pub use_bin_output: bool,
        pub deform: Deform,
        pub flipped: bool,
        pub dot: bool,
        pub angle: R64,
        pub grad_centr: bool,
        pub reverse: bool,
        pub coordconv: bool,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }

    impl ConvolutionalConfig {
        pub fn output_shape(&self, [h, w, _c]: [u64; 3]) -> [u64; 3] {
            let Self {
                filters,
                padding,
                size,
                stride_x,
                stride_y,
                ..
            } = *self;
            let out_h = (h + 2 * padding - size) / stride_y + 1;
            let out_w = (w + 2 * padding - size) / stride_x + 1;
            [out_h, out_w, filters]
        }
    }

    impl TryFrom<RawConvolutionalConfig> for ConvolutionalConfig {
        type Error = anyhow::Error;

        fn try_from(raw: RawConvolutionalConfig) -> Result<Self, Self::Error> {
            let RawConvolutionalConfig {
                filters,
                groups,
                size,
                stride,
                stride_x,
                stride_y,
                dilation,
                antialiasing,
                pad,
                padding,
                activation,
                assisted_excitation,
                share_index,
                batch_normalize,
                cbn,
                binary,
                xnor,
                use_bin_output,
                sway,
                rotate,
                stretch,
                stretch_sway,
                flipped,
                dot,
                angle,
                grad_centr,
                reverse,
                coordconv,
                common,
            } = raw;

            let stride_x = stride_x.unwrap_or(stride);
            let stride_y = stride_y.unwrap_or(stride);

            let padding = match (pad, padding) {
                (true, Some(_)) => {
                    warn!("padding option is ignored and is set to size / 2 due to pad == 1");
                    size / 2
                }
                (true, None) => size / 2,
                (false, padding) => padding.unwrap_or(0),
            };

            let deform = match (sway, rotate, stretch, stretch_sway) {
                (false, false, false, false) => Deform::None,
                (true, false, false, false) => Deform::Sway,
                (false, true, false, false) => Deform::Rotate,
                (false, false, true, false) => Deform::Stretch,
                (false, false, false, true) => Deform::StretchSway,
                _ => bail!("at most one of sway, rotate, stretch, stretch_sway can be set"),
            };

            // sanity check
            ensure!(
                size != 1 || dilation == 1,
                "dilation must be 1 if size is 1"
            );

            match (deform, size == 1) {
                (Deform::None, _) | (_, false) => (),
                (_, true) => {
                    bail!("sway, rotate, stretch, stretch_sway shoud be used with size >= 3")
                }
            }

            ensure!(!xnor || groups == 1, "groups must be 1 if xnor is enabled");

            Ok(Self {
                filters,
                groups,
                size,
                stride_x,
                stride_y,
                dilation,
                antialiasing,
                padding,
                activation,
                assisted_excitation,
                share_index,
                batch_normalize,
                cbn,
                binary,
                xnor,
                use_bin_output,
                deform,
                flipped,
                dot,
                angle,
                grad_centr,
                reverse,
                coordconv,
                common,
            })
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct RawConvolutionalConfig {
        pub filters: u64,
        #[serde(default = "defaults::groups")]
        pub groups: u64,
        pub size: u64,
        #[serde(default = "defaults::stride")]
        pub stride: u64,
        pub stride_x: Option<u64>,
        pub stride_y: Option<u64>,
        #[serde(default = "defaults::dilation")]
        pub dilation: u64,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub antialiasing: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub pad: bool,
        pub padding: Option<u64>,
        pub activation: Activation,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub assisted_excitation: bool,
        pub share_index: Option<LayerIndex>,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub batch_normalize: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub cbn: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub binary: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub xnor: bool,
        #[serde(
            rename = "bin_output",
            with = "serde_zero_one_bool",
            default = "defaults::bool_false"
        )]
        pub use_bin_output: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub sway: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub rotate: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub stretch: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub stretch_sway: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub flipped: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub dot: bool,
        #[serde(default = "defaults::angle")]
        pub angle: R64,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub grad_centr: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub reverse: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub coordconv: bool,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }

    impl From<ConvolutionalConfig> for RawConvolutionalConfig {
        fn from(conv: ConvolutionalConfig) -> Self {
            let ConvolutionalConfig {
                filters,
                groups,
                size,
                stride_x,
                stride_y,
                dilation,
                antialiasing,
                padding,
                activation,
                assisted_excitation,
                share_index,
                batch_normalize,
                cbn,
                binary,
                xnor,
                use_bin_output,
                deform,
                flipped,
                dot,
                angle,
                grad_centr,
                reverse,
                coordconv,
                common,
            } = conv;

            let (sway, rotate, stretch, stretch_sway) = match deform {
                Deform::None => (false, false, false, false),
                Deform::Sway => (true, false, false, false),
                Deform::Rotate => (false, true, false, false),
                Deform::Stretch => (false, false, true, false),
                Deform::StretchSway => (false, false, false, true),
            };

            Self {
                filters,
                groups,
                size,
                stride: defaults::stride(),
                stride_x: Some(stride_x),
                stride_y: Some(stride_y),
                dilation,
                antialiasing,
                pad: false,
                padding: Some(padding),
                activation,
                assisted_excitation,
                share_index,
                batch_normalize,
                cbn,
                binary,
                xnor,
                use_bin_output,
                sway,
                rotate,
                stretch,
                stretch_sway,
                flipped,
                dot,
                angle,
                grad_centr,
                reverse,
                coordconv,
                common,
            }
        }
    }
}

mod route_config {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    #[serde(try_from = "RawRouteConfig", into = "RawRouteConfig")]
    pub struct RouteConfig {
        #[derivative(Hash(hash_with = "hash_vec_layers"))]
        pub layers: IndexSet<LayerIndex>,
        pub group: RouteGroup,
        pub common: CommonLayerOptions,
    }

    impl TryFrom<RawRouteConfig> for RouteConfig {
        type Error = Error;

        fn try_from(from: RawRouteConfig) -> Result<Self, Self::Error> {
            let RawRouteConfig {
                layers,
                group_id,
                groups,
                common,
            } = from;

            let group = RouteGroup::new(group_id, groups.get())
                .ok_or_else(|| format_err!("group_id must be less than groups"))?;

            Ok(Self {
                layers,
                group,
                common,
            })
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    pub struct RawRouteConfig {
        #[derivative(Hash(hash_with = "hash_vec_layers"))]
        #[serde(with = "serde_vec_layers")]
        pub layers: IndexSet<LayerIndex>,
        #[serde(default = "defaults::route_groups")]
        pub groups: NonZeroU64,
        #[serde(default = "defaults::route_group_id")]
        pub group_id: u64,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }

    impl From<RouteConfig> for RawRouteConfig {
        fn from(from: RouteConfig) -> Self {
            let RouteConfig {
                layers,
                group,
                common,
            } = from;

            Self {
                layers,
                group_id: group.group_id(),
                groups: NonZeroU64::new(group.num_groups()).unwrap(),
                common,
            }
        }
    }
}

mod shortcut_config {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    pub struct ShortcutConfig {
        #[derivative(Hash(hash_with = "hash_vec_layers"))]
        #[serde(with = "serde_vec_layers")]
        pub from: IndexSet<LayerIndex>,
        pub activation: Activation,
        #[serde(with = "serde_weights_type", default = "defaults::weights_type")]
        pub weights_type: WeightsType,
        #[serde(default = "defaults::weights_normalization")]
        pub weights_normalization: WeightsNormalization,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }
}

mod max_pool_config {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(from = "RawMaxPoolConfig", into = "RawMaxPoolConfig")]
    pub struct MaxPoolConfig {
        pub stride_x: u64,
        pub stride_y: u64,
        pub size: u64,
        pub padding: u64,
        pub maxpool_depth: bool,
        pub out_channels: u64,
        pub antialiasing: bool,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }

    impl MaxPoolConfig {
        pub fn output_shape(&self, input_shape: [u64; 3]) -> [u64; 3] {
            let Self {
                padding,
                size,
                stride_x,
                stride_y,
                ..
            } = *self;
            let [in_h, in_w, in_c] = input_shape;

            let out_h = (in_h + padding - size) / stride_y + 1;
            let out_w = (in_w + padding - size) / stride_x + 1;
            let out_c = in_c;

            [out_h, out_w, out_c]
        }
    }

    impl From<RawMaxPoolConfig> for MaxPoolConfig {
        fn from(raw: RawMaxPoolConfig) -> Self {
            let RawMaxPoolConfig {
                stride,
                stride_x,
                stride_y,
                size,
                padding,
                maxpool_depth,
                out_channels,
                antialiasing,
                common,
            } = raw;

            let stride_x = stride_x.unwrap_or(stride);
            let stride_y = stride_y.unwrap_or(stride);
            let size = size.unwrap_or(stride);
            let padding = padding.unwrap_or(size - 1);

            Self {
                stride_x,
                stride_y,
                size,
                padding,
                maxpool_depth,
                out_channels,
                antialiasing,
                common,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct RawMaxPoolConfig {
        #[serde(default = "defaults::maxpool_stride")]
        pub stride: u64,
        pub stride_x: Option<u64>,
        pub stride_y: Option<u64>,
        pub size: Option<u64>,
        pub padding: Option<u64>,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub maxpool_depth: bool,
        #[serde(default = "defaults::out_channels")]
        pub out_channels: u64,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub antialiasing: bool,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }

    impl From<MaxPoolConfig> for RawMaxPoolConfig {
        fn from(maxpool: MaxPoolConfig) -> Self {
            let MaxPoolConfig {
                stride_x,
                stride_y,
                size,
                padding,
                maxpool_depth,
                out_channels,
                antialiasing,
                common,
            } = maxpool;

            Self {
                stride: defaults::maxpool_stride(),
                stride_x: Some(stride_x),
                stride_y: Some(stride_y),
                size: Some(size),
                padding: Some(padding),
                maxpool_depth,
                out_channels,
                antialiasing,
                common,
            }
        }
    }
}

mod up_sample_config {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct UpSampleConfig {
        #[serde(default = "defaults::upsample_stride")]
        pub stride: u64,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub reverse: bool,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }

    impl UpSampleConfig {
        pub fn output_shape(&self, input_shape: [u64; 3]) -> [u64; 3] {
            let Self {
                stride, reverse, ..
            } = *self;
            let [in_h, in_w, in_c] = input_shape;
            let (out_h, out_w) = if reverse {
                (in_h / stride, in_w / stride)
            } else {
                (in_h * stride, in_w * stride)
            };
            let out_c = in_c;
            [out_h, out_w, out_c]
        }
    }
}

mod batch_norm_config {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct BatchNormConfig {
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }
}

mod yolo_config {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    pub struct YoloConfig {
        pub max_boxes: u64,
        pub max_delta: Option<R64>,
        pub counters_per_class: Option<Vec<u64>>,
        pub label_smooth_eps: R64,
        pub scale_x_y: R64,
        pub objectness_smooth: bool,
        pub iou_normalizer: R64,
        pub obj_normalizer: R64,
        pub cls_normalizer: R64,
        pub delta_normalizer: R64,
        pub iou_thresh_kind: IouThreshold,
        pub beta_nms: R64,
        pub jitter: R64,
        pub resize: R64,
        pub focal_loss: bool,
        pub ignore_thresh: R64,
        pub truth_thresh: R64,
        pub iou_thresh: R64,
        pub random: R64,
        pub track_history_size: u64,
        pub sim_thresh: R64,
        pub dets_for_track: u64,
        pub dets_for_show: u64,
        pub track_ciou_norm: R64,
        pub embedding_layer: Option<LayerIndex>,
        pub map: Option<PathBuf>,
        pub anchors: Vec<(u64, u64)>,
        pub yolo_point: YoloPoint,
        pub iou_loss: IouLoss,
        pub nms_kind: NmsKind,
        pub common: CommonLayerOptions,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    pub struct RawYoloConfig {
        #[serde(default = "defaults::classes")]
        pub classes: u64,
        #[serde(default = "defaults::num")]
        pub num: u64,
        #[derivative(Hash(hash_with = "hash_option_vec_indexset::<u64, _>"))]
        #[serde(with = "serde_mask", default)]
        pub mask: Option<IndexSet<u64>>,
        #[serde(rename = "max", default = "defaults::max_boxes")]
        pub max_boxes: u64,
        pub max_delta: Option<R64>,
        #[serde(with = "serde_opt_vec_u64", default)]
        pub counters_per_class: Option<Vec<u64>>,
        #[serde(default = "defaults::yolo_label_smooth_eps")]
        pub label_smooth_eps: R64,
        #[serde(default = "defaults::scale_x_y")]
        pub scale_x_y: R64,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub objectness_smooth: bool,
        #[serde(default = "defaults::iou_normalizer")]
        pub iou_normalizer: R64,
        #[serde(default = "defaults::obj_normalizer")]
        pub obj_normalizer: R64,
        #[serde(default = "defaults::cls_normalizer")]
        pub cls_normalizer: R64,
        #[serde(default = "defaults::delta_normalizer")]
        pub delta_normalizer: R64,
        #[serde(default = "defaults::iou_loss")]
        pub iou_loss: IouLoss,
        #[serde(default = "defaults::iou_thresh_kind")]
        pub iou_thresh_kind: IouThreshold,
        #[serde(default = "defaults::beta_nms")]
        pub beta_nms: R64,
        #[serde(default = "defaults::nms_kind")]
        pub nms_kind: NmsKind,
        #[serde(default = "defaults::yolo_point")]
        pub yolo_point: YoloPoint,
        #[serde(default = "defaults::jitter")]
        pub jitter: R64,
        #[serde(default = "defaults::resize")]
        pub resize: R64,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub focal_loss: bool,
        #[serde(default = "defaults::ignore_thresh")]
        pub ignore_thresh: R64,
        #[serde(default = "defaults::truth_thresh")]
        pub truth_thresh: R64,
        #[serde(default = "defaults::iou_thresh")]
        pub iou_thresh: R64,
        #[serde(default = "defaults::random")]
        pub random: R64,
        #[serde(default = "defaults::track_history_size")]
        pub track_history_size: u64,
        #[serde(default = "defaults::sim_thresh")]
        pub sim_thresh: R64,
        #[serde(default = "defaults::dets_for_track")]
        pub dets_for_track: u64,
        #[serde(default = "defaults::dets_for_show")]
        pub dets_for_show: u64,
        #[serde(default = "defaults::track_ciou_norm")]
        pub track_ciou_norm: R64,
        pub embedding_layer: Option<LayerIndex>,
        pub map: Option<PathBuf>,
        #[serde(with = "serde_anchors", default)]
        pub anchors: Option<Vec<(u64, u64)>>,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }
}

mod gaussian_yolo_config {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    pub struct GaussianYoloConfig {
        pub max_boxes: u64,
        pub max_delta: Option<R64>,
        pub counters_per_class: Option<Vec<u64>>,
        pub label_smooth_eps: R64,
        pub scale_x_y: R64,
        pub objectness_smooth: bool,
        pub uc_normalizer: R64,
        pub iou_normalizer: R64,
        pub obj_normalizer: R64,
        pub cls_normalizer: R64,
        pub delta_normalizer: R64,
        pub iou_thresh_kind: IouThreshold,
        pub beta_nms: R64,
        pub jitter: R64,
        pub resize: R64,
        // pub focal_loss: bool,
        pub ignore_thresh: R64,
        pub truth_thresh: R64,
        pub iou_thresh: R64,
        pub random: R64,
        // pub track_history_size: u64,
        // pub sim_thresh: R64,
        // pub dets_for_track: u64,
        // pub dets_for_show: u64,
        // pub track_ciou_norm: R64,
        // pub embedding_layer: Option<LayerIndex>,
        pub map: Option<PathBuf>,
        pub anchors: Vec<(u64, u64)>,
        pub yolo_point: YoloPoint,
        pub iou_loss: IouLoss,
        pub nms_kind: NmsKind,
        pub common: CommonLayerOptions,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
    #[derivative(Hash)]
    pub struct RawGaussianYoloConfig {
        #[serde(default = "defaults::classes")]
        pub classes: u64,
        #[serde(rename = "max", default = "defaults::max_boxes")]
        pub max_boxes: u64,
        #[serde(default = "defaults::num")]
        pub num: u64,
        #[derivative(Hash(hash_with = "hash_option_vec_indexset::<u64, _>"))]
        #[serde(with = "serde_mask", default)]
        pub mask: Option<IndexSet<u64>>,
        pub max_delta: Option<R64>,
        #[serde(with = "serde_opt_vec_u64", default)]
        pub counters_per_class: Option<Vec<u64>>,
        #[serde(default = "defaults::yolo_label_smooth_eps")]
        pub label_smooth_eps: R64,
        #[serde(default = "defaults::scale_x_y")]
        pub scale_x_y: R64,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub objectness_smooth: bool,
        #[serde(default = "defaults::uc_normalizer")]
        pub uc_normalizer: R64,
        #[serde(default = "defaults::iou_normalizer")]
        pub iou_normalizer: R64,
        #[serde(default = "defaults::obj_normalizer")]
        pub obj_normalizer: R64,
        #[serde(default = "defaults::cls_normalizer")]
        pub cls_normalizer: R64,
        #[serde(default = "defaults::delta_normalizer")]
        pub delta_normalizer: R64,
        #[serde(default = "defaults::iou_loss")]
        pub iou_loss: IouLoss,
        #[serde(default = "defaults::iou_thresh_kind")]
        pub iou_thresh_kind: IouThreshold,
        #[serde(default = "defaults::beta_nms")]
        pub beta_nms: R64,
        #[serde(default = "defaults::nms_kind")]
        pub nms_kind: NmsKind,
        #[serde(default = "defaults::yolo_point")]
        pub yolo_point: YoloPoint,
        #[serde(default = "defaults::jitter")]
        pub jitter: R64,
        #[serde(default = "defaults::resize")]
        pub resize: R64,
        #[serde(default = "defaults::ignore_thresh")]
        pub ignore_thresh: R64,
        #[serde(default = "defaults::truth_thresh")]
        pub truth_thresh: R64,
        #[serde(default = "defaults::iou_thresh")]
        pub iou_thresh: R64,
        #[serde(default = "defaults::random")]
        pub random: R64,
        pub map: Option<PathBuf>,
        #[serde(with = "serde_anchors", default)]
        pub anchors: Option<Vec<(u64, u64)>>,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }
}

mod dropout_config {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(try_from = "RawDropoutConfig", into = "RawDropoutConfig")]
    pub struct DropoutConfig {
        pub probability: R64,
        pub dropblock: DropBlock,
        pub common: CommonLayerOptions,
    }

    impl TryFrom<RawDropoutConfig> for DropoutConfig {
        type Error = Error;

        fn try_from(from: RawDropoutConfig) -> Result<Self, Self::Error> {
            let RawDropoutConfig {
                probability,
                dropblock,
                dropblock_size_rel,
                dropblock_size_abs,
                common,
            } = from;

            let dropblock = match (dropblock, dropblock_size_rel, dropblock_size_abs) {
                (false, None, None) => DropBlock::None,
                (false, _, _) => bail!("neigher dropblock_size_rel nor dropblock_size_abs should be specified when dropblock is disabled"),
                (true, None, None) => bail!("dropblock is enabled, but none of dropblock_size_rel and dropblock_size_abs is specified"),
                (true, Some(val), None) => DropBlock::Relative(val),
                (true, None, Some(val)) => DropBlock::Absolute(val),
                (true, Some(_), Some(_)) => bail!("dropblock_size_rel and dropblock_size_abs cannot be specified together"),
            };

            Ok(Self {
                probability,
                dropblock,
                common,
            })
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct RawDropoutConfig {
        #[serde(default = "defaults::probability")]
        pub probability: R64,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub dropblock: bool,
        pub dropblock_size_rel: Option<R64>,
        pub dropblock_size_abs: Option<R64>,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }

    impl From<DropoutConfig> for RawDropoutConfig {
        fn from(from: DropoutConfig) -> Self {
            let DropoutConfig {
                probability,
                dropblock,
                common,
            } = from;

            let (dropblock, dropblock_size_rel, dropblock_size_abs) = match dropblock {
                DropBlock::None => (false, None, None),
                DropBlock::Relative(val) => (false, Some(val), None),
                DropBlock::Absolute(val) => (false, None, Some(val)),
            };

            Self {
                probability,
                dropblock,
                dropblock_size_rel,
                dropblock_size_abs,
                common,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum DropBlock {
        None,
        Absolute(R64),
        Relative(R64),
    }
}

mod softmax_config {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct SoftmaxConfig {
        pub groups: u64,
        pub temperature: R64,
        pub tree: Option<(PathBuf, Tree)>,
        pub spatial: R64,
        pub noloss: bool,
        pub common: CommonLayerOptions,
    }

    impl TryFrom<RawSoftmaxConfig> for SoftmaxConfig {
        type Error = Error;

        fn try_from(from: RawSoftmaxConfig) -> Result<Self, Self::Error> {
            let RawSoftmaxConfig {
                groups,
                temperature,
                tree_file,
                spatial,
                noloss,
                common,
            } = from;

            let tree = tree_file
                .map(|path| -> Result<_> {
                    unimplemented!();
                })
                .transpose()?;

            Ok(Self {
                groups,
                temperature,
                tree,
                spatial,
                noloss,
                common,
            })
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct RawSoftmaxConfig {
        #[serde(default = "defaults::softmax_groups")]
        pub groups: u64,
        #[serde(default = "defaults::temperature")]
        pub temperature: R64,
        pub tree_file: Option<PathBuf>,
        #[serde(default = "defaults::spatial")]
        pub spatial: R64,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub noloss: bool,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }

    impl TryFrom<SoftmaxConfig> for RawSoftmaxConfig {
        type Error = Error;

        fn try_from(from: SoftmaxConfig) -> Result<Self, Self::Error> {
            let SoftmaxConfig {
                groups,
                temperature,
                tree,
                spatial,
                noloss,
                common,
            } = from;

            let tree_file = tree
                .map(|(path, tree)| -> Result<_> {
                    unimplemented!();
                })
                .transpose()?;

            Ok(Self {
                groups,
                temperature,
                tree_file,
                spatial,
                noloss,
                common,
            })
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Tree {}
}

mod items {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum Item {
        #[serde(rename = "net")]
        Net(RawNetConfig),
        #[serde(rename = "connected")]
        Connected(ConnectedConfig),
        #[serde(rename = "convolutional")]
        Convolutional(ConvolutionalConfig),
        #[serde(rename = "route")]
        Route(RouteConfig),
        #[serde(rename = "shortcut")]
        Shortcut(ShortcutConfig),
        #[serde(rename = "maxpool")]
        MaxPool(MaxPoolConfig),
        #[serde(rename = "upsample")]
        UpSample(UpSampleConfig),
        #[serde(rename = "batchnorm")]
        BatchNorm(BatchNormConfig),
        #[serde(rename = "dropout")]
        Dropout(DropoutConfig),
        #[serde(rename = "softmax")]
        Softmax(RawSoftmaxConfig),
        #[serde(rename = "Gaussian_yolo")]
        GaussianYolo(RawGaussianYoloConfig),
        #[serde(rename = "yolo")]
        Yolo(RawYoloConfig),
    }

    impl TryFrom<DarknetConfig> for Items {
        type Error = Error;

        fn try_from(config: DarknetConfig) -> Result<Self, Self::Error> {
            let DarknetConfig {
                net: orig_net,
                layers: orig_layers,
            } = config;

            // extract global options that will be placed into yolo layers
            let (net, classes) = {
                let NetConfig {
                    max_batches,
                    batch,
                    learning_rate,
                    learning_rate_min,
                    sgdr_cycle,
                    sgdr_mult,
                    momentum,
                    decay,
                    subdivisions,
                    time_steps,
                    track,
                    augment_speed,
                    sequential_subdivisions,
                    try_fix_nan,
                    loss_scale,
                    dynamic_minibatch,
                    optimized_memory,
                    workspace_size_limit_mb,
                    adam,
                    input_size,
                    max_crop,
                    min_crop,
                    flip,
                    blur,
                    gaussian_noise,
                    mixup,
                    cutmux,
                    mosaic,
                    letter_box,
                    mosaic_bound,
                    contrastive,
                    contrastive_jit_flip,
                    contrastive_color,
                    unsupervised,
                    label_smooth_eps,
                    resize_step,
                    attention,
                    adversarial_lr,
                    max_chart_loss,
                    angle,
                    aspect,
                    saturation,
                    exposure,
                    hue,
                    power,
                    policy,
                    burn_in,
                    classes,
                } = orig_net;

                let (adam, b1, b2, eps) = match adam {
                    Some(Adam { b1, b2, eps }) => (true, b1, b2, eps),
                    None => (false, defaults::b1(), defaults::b2(), defaults::eps()),
                };

                let (inputs, height, width, channels) = match input_size {
                    Shape::Hwc([height, width, channels]) => {
                        (None, Some(height), Some(width), Some(channels))
                    }
                    Shape::Flat(inputs) => (Some(inputs), None, None, None),
                };

                let (policy, step, scale, steps, scales, seq_scales, gamma) = match policy {
                    Policy::Random => (
                        PolicyKind::Random,
                        defaults::step(),
                        defaults::scale(),
                        None,
                        None,
                        None,
                        defaults::gamma(),
                    ),
                    Policy::Poly => (
                        PolicyKind::Poly,
                        defaults::step(),
                        defaults::scale(),
                        None,
                        None,
                        None,
                        defaults::gamma(),
                    ),
                    Policy::Constant => (
                        PolicyKind::Constant,
                        defaults::step(),
                        defaults::scale(),
                        None,
                        None,
                        None,
                        defaults::gamma(),
                    ),
                    Policy::Step { step, scale } => (
                        PolicyKind::Step,
                        step,
                        scale,
                        None,
                        None,
                        None,
                        defaults::gamma(),
                    ),
                    Policy::Exp { gamma } => (
                        PolicyKind::Exp,
                        defaults::step(),
                        defaults::scale(),
                        None,
                        None,
                        None,
                        gamma,
                    ),
                    Policy::Sigmoid { gamma, step } => (
                        PolicyKind::Sigmoid,
                        step,
                        defaults::scale(),
                        None,
                        None,
                        None,
                        gamma,
                    ),
                    Policy::Steps {
                        steps,
                        scales,
                        seq_scales,
                    } => (
                        PolicyKind::Steps,
                        defaults::step(),
                        defaults::scale(),
                        Some(steps),
                        Some(scales),
                        Some(seq_scales),
                        defaults::gamma(),
                    ),
                    Policy::Sgdr => (
                        PolicyKind::Sgdr,
                        defaults::step(),
                        defaults::scale(),
                        None,
                        None,
                        None,
                        defaults::gamma(),
                    ),
                    Policy::SgdrCustom {
                        steps,
                        scales,
                        seq_scales,
                    } => (
                        PolicyKind::Sgdr,
                        defaults::step(),
                        defaults::scale(),
                        Some(steps),
                        Some(scales),
                        Some(seq_scales),
                        defaults::gamma(),
                    ),
                };

                let net = RawNetConfig {
                    max_batches,
                    batch,
                    learning_rate,
                    learning_rate_min,
                    sgdr_cycle: Some(sgdr_cycle),
                    sgdr_mult,
                    momentum,
                    decay,
                    subdivisions,
                    time_steps,
                    track,
                    augment_speed,
                    sequential_subdivisions: Some(sequential_subdivisions),
                    try_fix_nan,
                    loss_scale,
                    dynamic_minibatch,
                    optimized_memory,
                    workspace_size_limit_mb,
                    adam,
                    b1,
                    b2,
                    eps,
                    width: width.map(|w| NonZeroU64::new(w).unwrap()),
                    height: height.map(|h| NonZeroU64::new(h).unwrap()),
                    channels: channels.map(|c| NonZeroU64::new(c).unwrap()),
                    inputs: inputs.map(|i| NonZeroU64::new(i).unwrap()),
                    max_crop: Some(max_crop),
                    min_crop: Some(min_crop),
                    flip,
                    blur,
                    gaussian_noise,
                    mixup,
                    cutmux,
                    mosaic,
                    letter_box,
                    mosaic_bound,
                    contrastive,
                    contrastive_jit_flip,
                    contrastive_color,
                    unsupervised,
                    label_smooth_eps,
                    resize_step,
                    attention,
                    adversarial_lr,
                    max_chart_loss,
                    angle,
                    aspect,
                    saturation,
                    exposure,
                    hue,
                    power,
                    policy,
                    burn_in,
                    step,
                    scale,
                    steps,
                    scales,
                    seq_scales,
                    gamma,
                };

                (net, classes)
            };

            let global_anchors: Vec<_> = orig_layers
                .iter()
                .filter_map(|layer| match layer {
                    LayerConfig::Yolo(yolo) => {
                        let YoloConfig { anchors, .. } = yolo;
                        Some(anchors)
                    }
                    LayerConfig::GaussianYolo(yolo) => {
                        let GaussianYoloConfig { anchors, .. } = yolo;
                        Some(anchors)
                    }
                    _ => None,
                })
                .flat_map(|anchors| anchors.iter().cloned())
                .collect();

            let items: Vec<_> = {
                let mut mask_count = 0;

                iter::once(Ok(Item::Net(net)))
                    .chain(orig_layers.into_iter().map(|layer| -> Result<_> {
                        let item = match layer {
                            LayerConfig::Connected(layer) => Item::Connected(layer),
                            LayerConfig::Convolutional(layer) => Item::Convolutional(layer),
                            LayerConfig::Route(layer) => Item::Route(layer),
                            LayerConfig::Shortcut(layer) => Item::Shortcut(layer),
                            LayerConfig::MaxPool(layer) => Item::MaxPool(layer),
                            LayerConfig::UpSample(layer) => Item::UpSample(layer),
                            LayerConfig::BatchNorm(layer) => Item::BatchNorm(layer),
                            LayerConfig::Dropout(layer) => Item::Dropout(layer),
                            LayerConfig::Softmax(layer) => Item::Softmax(layer.try_into()?),
                            LayerConfig::Yolo(orig_layer) => {
                                let YoloConfig {
                                    max_boxes,
                                    max_delta,
                                    counters_per_class,
                                    label_smooth_eps,
                                    scale_x_y,
                                    objectness_smooth,
                                    iou_normalizer,
                                    obj_normalizer,
                                    cls_normalizer,
                                    delta_normalizer,
                                    iou_loss,
                                    iou_thresh_kind,
                                    beta_nms,
                                    nms_kind,
                                    yolo_point,
                                    jitter,
                                    resize,
                                    focal_loss,
                                    ignore_thresh,
                                    truth_thresh,
                                    iou_thresh,
                                    random,
                                    track_history_size,
                                    sim_thresh,
                                    dets_for_track,
                                    dets_for_show,
                                    track_ciou_norm,
                                    embedding_layer,
                                    map,
                                    anchors: local_anchors,
                                    common,
                                } = orig_layer;

                                // build mask list
                                let mask: IndexSet<_> = {
                                    let num_anchors = local_anchors.len();
                                    let mask_begin = mask_count;
                                    let mask_end = mask_begin + num_anchors;

                                    // update counter
                                    mask_count += num_anchors;

                                    (mask_begin..mask_end).map(|index| index as u64).collect()
                                };

                                // make sure mask indexes are valid
                                assert!(
                                    mask.iter()
                                        .cloned()
                                        .all(|index| (index as usize) < global_anchors.len()),
                                    "mask indexes must not exceed total number of anchors"
                                );

                                let num = global_anchors.len() as u64;
                                let mask = if mask.is_empty() { None } else { Some(mask) };
                                let anchors = if global_anchors.is_empty() {
                                    None
                                } else {
                                    Some(global_anchors.clone())
                                };

                                Item::Yolo(RawYoloConfig {
                                    classes,
                                    num,
                                    max_boxes,
                                    max_delta,
                                    counters_per_class,
                                    label_smooth_eps,
                                    scale_x_y,
                                    objectness_smooth,
                                    iou_normalizer,
                                    obj_normalizer,
                                    cls_normalizer,
                                    delta_normalizer,
                                    iou_loss,
                                    iou_thresh_kind,
                                    beta_nms,
                                    nms_kind,
                                    yolo_point,
                                    jitter,
                                    resize,
                                    focal_loss,
                                    ignore_thresh,
                                    truth_thresh,
                                    iou_thresh,
                                    random,
                                    track_history_size,
                                    sim_thresh,
                                    dets_for_track,
                                    dets_for_show,
                                    track_ciou_norm,
                                    embedding_layer,
                                    map,
                                    mask,
                                    anchors,
                                    common,
                                })
                            }
                            LayerConfig::GaussianYolo(orig_layer) => {
                                let GaussianYoloConfig {
                                    max_boxes,
                                    max_delta,
                                    counters_per_class,
                                    label_smooth_eps,
                                    scale_x_y,
                                    objectness_smooth,
                                    uc_normalizer,
                                    iou_normalizer,
                                    obj_normalizer,
                                    cls_normalizer,
                                    delta_normalizer,
                                    iou_loss,
                                    iou_thresh_kind,
                                    beta_nms,
                                    nms_kind,
                                    yolo_point,
                                    jitter,
                                    resize,
                                    ignore_thresh,
                                    truth_thresh,
                                    iou_thresh,
                                    random,
                                    map,
                                    anchors: local_anchors,
                                    common,
                                } = orig_layer;

                                // build mask list
                                let mask: IndexSet<_> = {
                                    let num_anchors = local_anchors.len();
                                    let mask_begin = mask_count;
                                    let mask_end = mask_begin + num_anchors;

                                    // update counter
                                    mask_count += num_anchors;

                                    (mask_begin..mask_end).map(|index| index as u64).collect()
                                };

                                // make sure mask indexes are valid
                                assert!(
                                    mask.iter()
                                        .cloned()
                                        .all(|index| (index as usize) < global_anchors.len()),
                                    "mask indexes must not exceed total number of anchors"
                                );

                                let num = global_anchors.len() as u64;
                                let mask = if mask.is_empty() { None } else { Some(mask) };
                                let anchors = if global_anchors.is_empty() {
                                    None
                                } else {
                                    Some(global_anchors.clone())
                                };

                                Item::GaussianYolo(RawGaussianYoloConfig {
                                    classes,
                                    num,
                                    max_boxes,
                                    max_delta,
                                    counters_per_class,
                                    label_smooth_eps,
                                    scale_x_y,
                                    objectness_smooth,
                                    uc_normalizer,
                                    iou_normalizer,
                                    obj_normalizer,
                                    cls_normalizer,
                                    delta_normalizer,
                                    iou_loss,
                                    iou_thresh_kind,
                                    beta_nms,
                                    nms_kind,
                                    yolo_point,
                                    jitter,
                                    resize,
                                    ignore_thresh,
                                    truth_thresh,
                                    iou_thresh,
                                    random,
                                    map,
                                    mask,
                                    anchors,
                                    common,
                                })
                            }
                        };

                        Ok(item)
                    }))
                    .try_collect()?
            };

            Ok(Items(items))
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(transparent)]
    pub struct Items(pub Vec<Item>);
}

mod misc {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct CommonLayerOptions {
        pub clip: Option<R64>,
        #[serde(
            rename = "onlyforward",
            with = "serde_zero_one_bool",
            default = "defaults::bool_false"
        )]
        pub only_forward: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub dont_update: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub burnin_update: bool,
        #[serde(
            rename = "stopbackward",
            with = "serde_zero_one_bool",
            default = "defaults::bool_false"
        )]
        pub stop_backward: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub train_only_bn: bool,
        #[serde(
            rename = "dontload",
            with = "serde_zero_one_bool",
            default = "defaults::bool_false"
        )]
        pub dont_load: bool,
        #[serde(
            rename = "dontloadscales",
            with = "serde_zero_one_bool",
            default = "defaults::bool_false"
        )]
        pub dont_load_scales: bool,
        #[serde(rename = "learning_rate", default = "defaults::learning_scale_scale")]
        pub learning_scale_scale: R64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum Deform {
        None,
        Sway,
        Rotate,
        Stretch,
        StretchSway,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum Activation {
        #[serde(rename = "mish")]
        Mish,
        #[serde(rename = "hard_mish")]
        HardMish,
        #[serde(rename = "swish")]
        Swish,
        #[serde(rename = "normalize_channels")]
        NormalizeChannels,
        #[serde(rename = "normalize_channels_softmax")]
        NormalizeChannelsSoftmax,
        #[serde(rename = "normalize_channels_softmax_maxval")]
        NormalizeChannelsSoftmaxMaxval,
        #[serde(rename = "logistic")]
        Logistic,
        #[serde(rename = "loggy")]
        Loggy,
        #[serde(rename = "relu")]
        Relu,
        #[serde(rename = "elu")]
        Elu,
        #[serde(rename = "selu")]
        Selu,
        #[serde(rename = "gelu")]
        Gelu,
        #[serde(rename = "relie")]
        Relie,
        #[serde(rename = "ramp")]
        Ramp,
        #[serde(rename = "linear")]
        Linear,
        #[serde(rename = "tanh")]
        Tanh,
        #[serde(rename = "plse")]
        Plse,
        #[serde(rename = "leaky")]
        Leaky,
        #[serde(rename = "stair")]
        Stair,
        #[serde(rename = "hardtan")]
        Hardtan,
        #[serde(rename = "lhtan")]
        Lhtan,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum IouLoss {
        #[serde(rename = "mse")]
        Mse,
        #[serde(rename = "iou")]
        IoU,
        #[serde(rename = "giou")]
        GIoU,
        #[serde(rename = "diou")]
        DIoU,
        #[serde(rename = "ciou")]
        CIoU,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum IouThreshold {
        #[serde(rename = "iou")]
        IoU,
        #[serde(rename = "giou")]
        GIoU,
        #[serde(rename = "diou")]
        DIoU,
        #[serde(rename = "ciou")]
        CIoU,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum YoloPoint {
        #[serde(rename = "center")]
        Center,
        #[serde(rename = "left_top")]
        LeftTop,
        #[serde(rename = "right_bottom")]
        RightBottom,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum NmsKind {
        #[serde(rename = "default")]
        Default,
        #[serde(rename = "greedynms")]
        Greedy,
        #[serde(rename = "diounms")]
        DIoU,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum PolicyKind {
        #[serde(rename = "random")]
        Random,
        #[serde(rename = "poly")]
        Poly,
        #[serde(rename = "constant")]
        Constant,
        #[serde(rename = "step")]
        Step,
        #[serde(rename = "exp")]
        Exp,
        #[serde(rename = "sigmoid")]
        Sigmoid,
        #[serde(rename = "steps")]
        Steps,
        #[serde(rename = "sgdr")]
        Sgdr,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum Policy {
        Random,
        Poly,
        Constant,
        Step {
            step: u64,
            scale: R64,
        },
        Exp {
            gamma: R64,
        },
        Sigmoid {
            gamma: R64,
            step: u64,
        },
        Steps {
            steps: Vec<u64>,
            scales: Vec<R64>,
            seq_scales: Vec<R64>,
        },
        Sgdr,
        SgdrCustom {
            steps: Vec<u64>,
            scales: Vec<R64>,
            seq_scales: Vec<R64>,
        },
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize_repr, Deserialize_repr)]
    #[repr(u64)]
    pub enum MixUp {
        MixUp = 1,
        CutMix = 2,
        Mosaic = 3,
        Random = 4,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum LayerIndex {
        Relative(NonZeroUsize),
        Absolute(usize),
    }

    impl LayerIndex {
        pub fn relative(&self) -> Option<usize> {
            match *self {
                Self::Relative(index) => Some(index.get()),
                Self::Absolute(_) => None,
            }
        }

        pub fn absolute(&self) -> Option<usize> {
            match *self {
                Self::Absolute(index) => Some(index),
                Self::Relative(_) => None,
            }
        }

        pub fn to_absolute(&self, curr_index: usize) -> Option<usize> {
            match *self {
                Self::Absolute(index) => Some(index),
                Self::Relative(index) => {
                    let index = index.get();
                    if index <= curr_index {
                        Some(curr_index - index)
                    } else {
                        None
                    }
                }
            }
        }
    }

    impl From<isize> for LayerIndex {
        fn from(index: isize) -> Self {
            if index < 0 {
                Self::Relative(NonZeroUsize::new(-index as usize).unwrap())
            } else {
                Self::Absolute(index as usize)
            }
        }
    }

    impl From<LayerIndex> for isize {
        fn from(index: LayerIndex) -> Self {
            match index {
                LayerIndex::Relative(index) => -(index.get() as isize),
                LayerIndex::Absolute(index) => index as isize,
            }
        }
    }

    impl Serialize for LayerIndex {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            isize::from(*self).serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for LayerIndex {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let index: Self = isize::deserialize(deserializer)?.into();
            Ok(index)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum Shape {
        Hwc([u64; 3]),
        Flat(u64),
    }

    impl Shape {
        pub fn to_vec(&self) -> Vec<u64> {
            match *self {
                Self::Hwc(hwc) => Vec::from(hwc),
                Self::Flat(flat) => vec![flat],
            }
        }

        pub fn hwc(&self) -> Option<[u64; 3]> {
            match *self {
                Self::Hwc(hwc) => Some(hwc),
                Self::Flat(_) => None,
            }
        }

        pub fn flat(&self) -> Option<u64> {
            match *self {
                Self::Flat(flat) => Some(flat),
                Self::Hwc(_) => None,
            }
        }
    }

    impl Display for Shape {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Hwc([h, w, c]) => f.debug_list().entries(vec![h, w, c]).finish(),
                Self::Flat(size) => write!(f, "{}", size),
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Adam {
        pub b1: R64,
        pub b2: R64,
        pub eps: R64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum WeightsType {
        #[serde(rename = "none")]
        None,
        #[serde(rename = "per_feature")]
        PerFeature,
        #[serde(rename = "per_channel")]
        PerChannel,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum WeightsNormalization {
        #[serde(rename = "none")]
        None,
        #[serde(rename = "relu")]
        Relu,
        #[serde(rename = "softmax")]
        Softmax,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct RouteGroup {
        group_id: u64,
        num_groups: u64,
    }

    impl RouteGroup {
        pub fn new(group_id: u64, num_groups: u64) -> Option<Self> {
            if num_groups == 0 || group_id >= num_groups {
                None
            } else {
                Some(Self {
                    group_id,
                    num_groups,
                })
            }
        }

        pub fn group_id(&self) -> u64 {
            self.group_id
        }

        pub fn num_groups(&self) -> u64 {
            self.num_groups
        }
    }
}

// utility functions

mod defaults {
    use super::*;

    pub fn bool_true() -> bool {
        true
    }

    pub fn bool_false() -> bool {
        false
    }

    pub fn max_batches() -> u64 {
        0
    }

    pub fn batch() -> u64 {
        1
    }

    pub fn learning_rate() -> R64 {
        R64::new(0.001)
    }

    pub fn learning_rate_min() -> R64 {
        R64::new(0.00001)
    }

    pub fn sgdr_mult() -> u64 {
        2
    }

    pub fn momentum() -> R64 {
        R64::new(0.9)
    }

    pub fn decay() -> R64 {
        R64::new(0.0001)
    }

    pub fn subdivisions() -> u64 {
        1
    }

    pub fn time_steps() -> u64 {
        1
    }

    pub fn track() -> u64 {
        1
    }

    pub fn augment_speed() -> u64 {
        2
    }

    pub fn workspace_size_limit_mb() -> u64 {
        1024
    }

    pub fn b1() -> R64 {
        R64::new(0.9)
    }

    pub fn b2() -> R64 {
        R64::new(0.999)
    }

    pub fn eps() -> R64 {
        R64::new(0.000001)
    }

    pub fn groups() -> u64 {
        1
    }

    pub fn stride() -> u64 {
        1
    }

    pub fn dilation() -> u64 {
        1
    }

    pub fn angle() -> R64 {
        R64::new(15.0)
    }

    pub fn loss_scale() -> R64 {
        R64::new(1.0)
    }

    pub fn mixup() -> MixUp {
        MixUp::Random
    }

    pub fn label_smooth_eps() -> R64 {
        R64::new(0.0)
    }

    pub fn resize_step() -> u64 {
        32
    }

    pub fn adversarial_lr() -> R64 {
        R64::new(0.0)
    }

    pub fn max_chart_loss() -> R64 {
        R64::new(20.0)
    }

    pub fn aspect() -> R64 {
        R64::new(1.0)
    }

    pub fn saturation() -> R64 {
        R64::new(1.0)
    }

    pub fn exposure() -> R64 {
        R64::new(1.0)
    }

    pub fn hue() -> R64 {
        R64::new(0.0)
    }

    pub fn power() -> R64 {
        R64::new(4.0)
    }

    pub fn policy() -> PolicyKind {
        PolicyKind::Constant
    }

    pub fn step() -> u64 {
        1
    }

    pub fn scale() -> R64 {
        R64::new(1.0)
    }

    pub fn gamma() -> R64 {
        R64::new(1.0)
    }

    pub fn burn_in() -> u64 {
        0
    }

    pub fn route_groups() -> NonZeroU64 {
        NonZeroU64::new(1).unwrap()
    }

    pub fn route_group_id() -> u64 {
        0
    }

    pub fn weights_type() -> WeightsType {
        WeightsType::None
    }

    pub fn weights_normalization() -> WeightsNormalization {
        WeightsNormalization::None
    }

    pub fn maxpool_stride() -> u64 {
        1
    }

    pub fn out_channels() -> u64 {
        1
    }

    pub fn upsample_stride() -> u64 {
        2
    }

    pub fn classes() -> u64 {
        warn!("classes option is not specified, use default 20");
        20
    }

    pub fn num() -> u64 {
        1
    }

    pub fn max_boxes() -> u64 {
        200
    }

    pub fn yolo_label_smooth_eps() -> R64 {
        R64::new(0.0)
    }

    pub fn scale_x_y() -> R64 {
        R64::new(1.0)
    }

    pub fn uc_normalizer() -> R64 {
        R64::new(1.0)
    }

    pub fn iou_normalizer() -> R64 {
        R64::new(0.75)
    }

    pub fn obj_normalizer() -> R64 {
        R64::new(1.0)
    }

    pub fn cls_normalizer() -> R64 {
        R64::new(1.0)
    }

    pub fn delta_normalizer() -> R64 {
        R64::new(1.0)
    }

    pub fn iou_loss() -> IouLoss {
        IouLoss::Mse
    }

    pub fn iou_thresh_kind() -> IouThreshold {
        IouThreshold::IoU
    }

    pub fn beta_nms() -> R64 {
        R64::new(0.6)
    }

    pub fn nms_kind() -> NmsKind {
        NmsKind::Default
    }

    pub fn yolo_point() -> YoloPoint {
        YoloPoint::Center
    }

    pub fn jitter() -> R64 {
        R64::new(0.2)
    }

    pub fn resize() -> R64 {
        R64::new(1.0)
    }

    pub fn ignore_thresh() -> R64 {
        R64::new(0.5)
    }

    pub fn truth_thresh() -> R64 {
        R64::new(1.0)
    }

    pub fn iou_thresh() -> R64 {
        R64::new(1.0)
    }

    pub fn random() -> R64 {
        R64::new(0.0)
    }

    pub fn track_history_size() -> u64 {
        5
    }

    pub fn sim_thresh() -> R64 {
        R64::new(0.8)
    }

    pub fn dets_for_track() -> u64 {
        1
    }

    pub fn dets_for_show() -> u64 {
        1
    }

    pub fn track_ciou_norm() -> R64 {
        R64::new(0.01)
    }

    pub fn connected_output() -> u64 {
        1
    }

    pub fn connected_activation() -> Activation {
        Activation::Logistic
    }

    pub fn learning_scale_scale() -> R64 {
        R64::new(1.0)
    }

    pub fn probability() -> R64 {
        R64::new(0.2)
    }

    pub fn softmax_groups() -> u64 {
        1
    }

    pub fn temperature() -> R64 {
        R64::new(1.0)
    }

    pub fn spatial() -> R64 {
        R64::new(0.0)
    }
}

fn hash_vec_layers<H>(layers: &IndexSet<LayerIndex>, state: &mut H)
where
    H: Hasher,
{
    let layers: Vec<_> = layers.iter().cloned().collect();
    layers.hash(state);
}

fn hash_vec_indexset<T, H>(set: &IndexSet<T>, state: &mut H)
where
    T: Hash,
    H: Hasher,
{
    let set: Vec<_> = set.iter().collect();
    set.hash(state);
}

fn hash_option_vec_indexset<T, H>(opt: &Option<IndexSet<T>>, state: &mut H)
where
    T: Hash,
    H: Hasher,
{
    let opt: Option<Vec<_>> = opt.as_ref().map(|set| set.iter().collect());
    opt.hash(state);
}

mod serde_zero_one_bool {
    use super::*;

    pub fn serialize<S>(&yes: &bool, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (yes as i64).serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<bool, D::Error>
    where
        D: Deserializer<'de>,
    {
        match i64::deserialize(deserializer)? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(D::Error::invalid_value(
                de::Unexpected::Signed(value),
                &"0 or 1",
            )),
        }
    }
}

mod serde_vec_layers {
    use super::*;

    pub fn serialize<S>(indexes: &IndexSet<LayerIndex>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let text = indexes
            .iter()
            .cloned()
            .map(|index| isize::from(index).to_string())
            .join(",");
        text.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<IndexSet<LayerIndex>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        let layers_vec: Vec<_> = text
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            .split(",")
            .map(|token| -> Result<_> {
                let index: isize = token.parse()?;
                let index = LayerIndex::from(index);
                Ok(index)
            })
            .try_collect()
            .map_err(|err| D::Error::custom(format!("failed to parse layer index: {:?}", err)))?;
        let layers_set: IndexSet<LayerIndex> = layers_vec.iter().cloned().collect();

        if layers_vec.len() != layers_set.len() {
            return Err(D::Error::custom("duplicated layer index is not allowed"));
        }

        Ok(layers_set)
    }
}

mod serde_mask {
    use super::*;

    pub fn serialize<S>(steps: &Option<IndexSet<u64>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        steps
            .as_ref()
            .map(|steps| steps.iter().map(|step| step.to_string()).join(","))
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<IndexSet<u64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        let steps_vec: Vec<u64> = text
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            .split(",")
            .map(|token| token.parse())
            .try_collect()
            .map_err(|err| D::Error::custom(format!("failed to parse steps: {:?}", err)))?;
        let steps_set: IndexSet<_> = steps_vec.iter().cloned().collect();

        if steps_vec.len() != steps_set.len() {
            return Err(D::Error::custom("duplicated mask indexes is not allowed"));
        }

        Ok(Some(steps_set))
    }
}

mod serde_opt_vec_u64 {
    use super::*;

    pub fn serialize<S>(steps: &Option<Vec<u64>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        steps
            .as_ref()
            .map(|steps| steps.iter().map(|step| step.to_string()).join(","))
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Vec<u64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = <Option<String>>::deserialize(deserializer)?;
        let steps: Option<Vec<u64>> = text
            .map(|text| {
                text.chars()
                    .filter(|c| !c.is_whitespace())
                    .collect::<String>()
                    .split(",")
                    .map(|token| token.parse())
                    .try_collect()
            })
            .transpose()
            .map_err(|err| D::Error::custom(format!("failed to parse steps: {:?}", err)))?;
        Ok(steps)
    }
}

mod serde_opt_vec_r64 {
    use super::*;

    pub fn serialize<S>(scales: &Option<Vec<R64>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        scales
            .as_ref()
            .map(|steps| steps.iter().map(|step| step.to_string()).join(","))
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Vec<R64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = <Option<String>>::deserialize(deserializer)?;
        let scales: Option<Vec<R64>> = text
            .map(|text| {
                text.chars()
                    .filter(|c| !c.is_whitespace())
                    .collect::<String>()
                    .split(",")
                    .map(|token| {
                        let value: f64 = token.parse().map_err(|err| {
                            D::Error::custom(format!("failed to parse steps: {:?}", err))
                        })?;
                        let value = R64::try_new(value).ok_or_else(|| {
                            D::Error::custom(format!("invalid value '{}'", token))
                        })?;
                        Ok(value)
                    })
                    .try_collect()
            })
            .transpose()?;
        Ok(scales)
    }
}

mod serde_anchors {
    use super::*;

    pub fn serialize<S>(steps: &Option<Vec<(u64, u64)>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        steps
            .as_ref()
            .map(|steps| {
                steps
                    .iter()
                    .flat_map(|(w, h)| vec![w, h])
                    .map(|val| val.to_string())
                    .join(",")
            })
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Vec<(u64, u64)>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = match Option::<String>::deserialize(deserializer)? {
            Some(text) => text,
            None => return Ok(None),
        };
        let values: Vec<u64> = text
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            .split(",")
            .map(|token| token.parse())
            .try_collect()
            .map_err(|err| D::Error::custom(format!("failed to parse anchors: {:?}", err)))?;

        if values.len() % 2 != 0 {
            return Err(D::Error::custom("expect even number of values"));
        }

        let anchors: Vec<_> = values
            .into_iter()
            .chunks(2)
            .into_iter()
            .map(|mut chunk| (chunk.next().unwrap(), chunk.next().unwrap()))
            .collect();
        Ok(Some(anchors))
    }
}

mod serde_weights_type {
    use super::*;

    pub fn serialize<S>(weights_type: &WeightsType, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match weights_type {
            WeightsType::None => "none",
            WeightsType::PerFeature => "per_feature",
            WeightsType::PerChannel => "per_channel",
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<WeightsType, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        let weights_type = match text.as_str() {
            "per_feature" | "per_layer" => WeightsType::PerFeature,
            "per_channel" => WeightsType::PerChannel,
            _ => {
                return Err(D::Error::custom(format!(
                    "'{}' is not a valid weights type",
                    text
                )))
            }
        };
        Ok(weights_type)
    }
}
