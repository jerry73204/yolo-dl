use crate::common::*;

pub use items::*;

pub trait CommonLayerOptionsEx {
    fn common(&self) -> &CommonLayerOptions;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "Vec<Item>", into = "Vec<Item>")]
pub struct Config {
    pub net: Net,
    pub layers: Vec<Layer>,
}

impl TryFrom<Vec<Item>> for Config {
    type Error = anyhow::Error;

    fn try_from(items: Vec<Item>) -> Result<Self, Self::Error> {
        let mut items = items.into_iter();
        let net = {
            let first = items.next().ok_or_else(|| format_err!("no items found"))?;
            match first {
                Item::Net(net) => net,
                _ => bail!("the first item must be 'net'"),
            }
        };
        let layers: Vec<_> = items
            .map(|item| {
                let layer = match item {
                    Item::Connected(layer) => Layer::Connected(layer),
                    Item::Convolutional(layer) => Layer::Convolutional(layer),
                    Item::Route(layer) => Layer::Route(layer),
                    Item::Shortcut(layer) => Layer::Shortcut(layer),
                    Item::MaxPool(layer) => Layer::MaxPool(layer),
                    Item::UpSample(layer) => Layer::UpSample(layer),
                    Item::Yolo(layer) => Layer::Yolo(layer),
                    Item::BatchNorm(layer) => Layer::BatchNorm(layer),
                    Item::Net(_layer) => bail!("the 'net' layer must appear in the first section"),
                };
                Ok(layer)
            })
            .try_collect()?;

        Ok(Config { net, layers })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Layer {
    #[serde(rename = "connected")]
    Connected(Connected),
    #[serde(rename = "convolutional")]
    Convolutional(Convolutional),
    #[serde(rename = "route")]
    Route(Route),
    #[serde(rename = "shortcut")]
    Shortcut(Shortcut),
    #[serde(rename = "maxpool")]
    MaxPool(MaxPool),
    #[serde(rename = "upsample")]
    UpSample(UpSample),
    #[serde(rename = "yolo")]
    Yolo(Yolo),
    #[serde(rename = "batchnorm")]
    BatchNorm(BatchNorm),
}

impl CommonLayerOptionsEx for Layer {
    fn common(&self) -> &CommonLayerOptions {
        match self {
            Layer::Connected(layer) => layer.common(),
            Layer::Convolutional(layer) => layer.common(),
            Layer::Route(layer) => layer.common(),
            Layer::Shortcut(layer) => layer.common(),
            Layer::MaxPool(layer) => layer.common(),
            Layer::UpSample(layer) => layer.common(),
            Layer::Yolo(layer) => layer.common(),
            Layer::BatchNorm(layer) => layer.common(),
        }
    }
}

impl From<Config> for Vec<Item> {
    fn from(config: Config) -> Self {
        let Config { net, layers } = config;
        let items: Vec<_> = iter::once(Item::Net(net))
            .chain(layers.into_iter().map(|layer| match layer {
                Layer::Connected(layer) => Item::Connected(layer),
                Layer::Convolutional(layer) => Item::Convolutional(layer),
                Layer::Route(layer) => Item::Route(layer),
                Layer::Shortcut(layer) => Item::Shortcut(layer),
                Layer::MaxPool(layer) => Item::MaxPool(layer),
                Layer::UpSample(layer) => Item::UpSample(layer),
                Layer::Yolo(layer) => Item::Yolo(layer),
                Layer::BatchNorm(layer) => Item::BatchNorm(layer),
            }))
            .collect();
        items
    }
}

mod items {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum Item {
        #[serde(rename = "net")]
        Net(Net),
        #[serde(rename = "connected")]
        Connected(Connected),
        #[serde(rename = "convolutional")]
        Convolutional(Convolutional),
        #[serde(rename = "route")]
        Route(Route),
        #[serde(rename = "shortcut")]
        Shortcut(Shortcut),
        #[serde(rename = "maxpool")]
        MaxPool(MaxPool),
        #[serde(rename = "upsample")]
        UpSample(UpSample),
        #[serde(rename = "yolo")]
        Yolo(Yolo),
        #[serde(rename = "batchnorm")]
        BatchNorm(BatchNorm),
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(try_from = "RawNet", into = "RawNet")]
    pub struct Net {
        pub max_batches: usize,
        pub batch: usize,
        pub learning_rate: R64,
        pub learning_rate_min: R64,
        pub sgdr_cycle: usize,
        pub sgdr_mult: usize,
        pub momentum: R64,
        pub decay: R64,
        pub subdivisions: usize,
        pub time_steps: usize,
        pub track: usize,
        pub augment_speed: usize,
        pub sequential_subdivisions: usize,
        pub try_fix_nan: bool,
        pub loss_scale: R64,
        pub dynamic_minibatch: bool,
        pub optimized_memory: bool,
        pub workspace_size_limit_mb: usize,
        pub adam: Option<Adam>,
        pub input_size: Shape,
        pub max_crop: usize,
        pub min_crop: usize,
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
        pub resize_step: usize,
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
        pub burn_in: usize,
    }

    impl TryFrom<RawNet> for Net {
        type Error = Error;

        fn try_from(raw: RawNet) -> Result<Self, Self::Error> {
            let RawNet {
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
            } = raw;

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
                    let steps = steps
                        .ok_or_else(|| format_err!("steps must be specified for step policy"))?;
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

            Ok(Net {
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
            })
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct RawNet {
        #[serde(default = "defaults::max_batches")]
        pub max_batches: usize,
        #[serde(default = "defaults::batch")]
        pub batch: usize,
        #[serde(default = "defaults::learning_rate")]
        pub learning_rate: R64,
        #[serde(default = "defaults::learning_rate_min")]
        pub learning_rate_min: R64,
        pub sgdr_cycle: Option<usize>,
        #[serde(default = "defaults::sgdr_mult")]
        pub sgdr_mult: usize,
        #[serde(default = "defaults::momentum")]
        pub momentum: R64,
        #[serde(default = "defaults::decay")]
        pub decay: R64,
        #[serde(default = "defaults::subdivisions")]
        pub subdivisions: usize,
        #[serde(default = "defaults::time_steps")]
        pub time_steps: usize,
        #[serde(default = "defaults::track")]
        pub track: usize,
        #[serde(default = "defaults::augment_speed")]
        pub augment_speed: usize,
        pub sequential_subdivisions: Option<usize>,
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
        pub workspace_size_limit_mb: usize,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub adam: bool,
        #[serde(rename = "B1", default = "defaults::b1")]
        pub b1: R64,
        #[serde(rename = "B2", default = "defaults::b2")]
        pub b2: R64,
        #[serde(default = "defaults::eps")]
        pub eps: R64,
        pub width: Option<NonZeroUsize>,
        pub height: Option<NonZeroUsize>,
        pub channels: Option<NonZeroUsize>,
        pub inputs: Option<NonZeroUsize>,
        pub max_crop: Option<usize>,
        pub min_crop: Option<usize>,
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
        pub resize_step: usize,
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
        pub burn_in: usize,
        #[serde(default = "defaults::step")]
        pub step: usize,
        #[serde(default = "defaults::scale")]
        pub scale: R64,
        #[serde(with = "serde_opt_vec_usize", default)]
        pub steps: Option<Vec<usize>>,
        #[serde(with = "serde_opt_vec_r64", default)]
        pub scales: Option<Vec<R64>>,
        #[serde(with = "serde_opt_vec_r64", default)]
        pub seq_scales: Option<Vec<R64>>,
        #[serde(default = "defaults::gamma")]
        pub gamma: R64,
    }

    impl From<Net> for RawNet {
        fn from(net: Net) -> Self {
            let Net {
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
            } = net;

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

            RawNet {
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
                width: width.map(|w| NonZeroUsize::new(w).unwrap()),
                height: height.map(|h| NonZeroUsize::new(h).unwrap()),
                channels: channels.map(|c| NonZeroUsize::new(c).unwrap()),
                inputs: inputs.map(|i| NonZeroUsize::new(i).unwrap()),
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
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Connected {
        #[serde(default = "defaults::connected_output")]
        pub output: usize,
        #[serde(default = "defaults::connected_activation")]
        pub activation: Activation,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub batch_normalize: bool,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }

    impl CommonLayerOptionsEx for Connected {
        fn common(&self) -> &CommonLayerOptions {
            &self.common
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(try_from = "RawConvolutional", into = "RawConvolutional")]
    pub struct Convolutional {
        pub filters: usize,
        pub groups: usize,
        pub size: usize,
        pub batch_normalize: bool,
        pub stride_x: usize,
        pub stride_y: usize,
        pub dilation: usize,
        pub antialiasing: bool,
        pub padding: usize,
        pub activation: Activation,
        pub assisted_excitation: bool,
        pub share_index: Option<usize>,
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

    impl Convolutional {
        pub fn num_weights(&self, channels: usize) -> Result<usize> {
            ensure!(
                channels % self.groups == 0,
                "the input channels is not multiple of groups"
            );
            Ok((channels / self.groups) * self.filters * self.size.pow(2))
        }
    }

    impl CommonLayerOptionsEx for Convolutional {
        fn common(&self) -> &CommonLayerOptions {
            &self.common
        }
    }

    impl TryFrom<RawConvolutional> for Convolutional {
        type Error = anyhow::Error;

        fn try_from(raw: RawConvolutional) -> Result<Self, Self::Error> {
            let RawConvolutional {
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
    pub struct RawConvolutional {
        pub filters: usize,
        #[serde(default = "defaults::groups")]
        pub groups: usize,
        pub size: usize,
        #[serde(default = "defaults::stride")]
        pub stride: usize,
        pub stride_x: Option<usize>,
        pub stride_y: Option<usize>,
        #[serde(default = "defaults::dilation")]
        pub dilation: usize,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub antialiasing: bool,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub pad: bool,
        pub padding: Option<usize>,
        pub activation: Activation,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub assisted_excitation: bool,
        pub share_index: Option<usize>,
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

    impl From<Convolutional> for RawConvolutional {
        fn from(conv: Convolutional) -> Self {
            let Convolutional {
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

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Route {
        #[serde(with = "serde_vec_isize")]
        pub layers: Vec<isize>,
        #[serde(default = "defaults::route_groups")]
        pub groups: usize,
        #[serde(default = "defaults::route_group_id")]
        pub groupd_id: usize,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }

    impl CommonLayerOptionsEx for Route {
        fn common(&self) -> &CommonLayerOptions {
            &self.common
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Shortcut {
        #[serde(with = "serde_vec_isize", default)]
        pub from: Vec<isize>,
        pub activation: Activation,
        #[serde(with = "serde_weights_type", default = "defaults::weights_type")]
        pub weights_type: WeightsType,
        #[serde(default = "defaults::weights_normalization")]
        pub weights_normalization: WeightsNormalization,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }

    impl Shortcut {
        pub fn num_weights(&self, channels: usize) -> usize {
            match self.weights_type {
                WeightsType::None => 0,
                WeightsType::PerFeature => self.from.len() + 1,
                WeightsType::PerChannel => (self.from.len() + 1) * channels,
            }
        }
    }

    impl CommonLayerOptionsEx for Shortcut {
        fn common(&self) -> &CommonLayerOptions {
            &self.common
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(from = "RawMaxPool", into = "RawMaxPool")]
    pub struct MaxPool {
        pub stride_x: usize,
        pub stride_y: usize,
        pub size: usize,
        pub padding: usize,
        pub maxpool_depth: usize,
        pub out_channels: usize,
        pub antialiasing: bool,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }

    impl CommonLayerOptionsEx for MaxPool {
        fn common(&self) -> &CommonLayerOptions {
            &self.common
        }
    }

    impl From<RawMaxPool> for MaxPool {
        fn from(raw: RawMaxPool) -> Self {
            let RawMaxPool {
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
    pub struct RawMaxPool {
        #[serde(default = "defaults::maxpool_stride")]
        pub stride: usize,
        pub stride_x: Option<usize>,
        pub stride_y: Option<usize>,
        pub size: Option<usize>,
        pub padding: Option<usize>,
        #[serde(default = "defaults::maxpool_depth")]
        pub maxpool_depth: usize,
        #[serde(default = "defaults::out_channels")]
        pub out_channels: usize,
        #[serde(with = "serde_zero_one_bool", default = "defaults::bool_false")]
        pub antialiasing: bool,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }

    impl From<MaxPool> for RawMaxPool {
        fn from(maxpool: MaxPool) -> Self {
            let MaxPool {
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

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct UpSample {
        #[serde(default = "defaults::upsample_stride")]
        pub stride: usize,
        #[serde(default = "defaults::upsample_scale")]
        pub scale: usize,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }

    impl CommonLayerOptionsEx for UpSample {
        fn common(&self) -> &CommonLayerOptions {
            &self.common
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Yolo {
        #[serde(default = "defaults::classes")]
        pub classes: usize,
        #[serde(default = "defaults::num")]
        pub num: usize,
        #[serde(with = "serde_vec_usize")]
        pub mask: Vec<usize>,
        #[serde(rename = "max", default = "defaults::max_boxes")]
        pub max_boxes: usize,
        pub max_delta: Option<R64>,
        #[serde(with = "serde_opt_vec_usize", default)]
        pub counters_per_class: Option<Vec<usize>>,
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
        pub track_history_size: usize,
        #[serde(default = "defaults::sim_thresh")]
        pub sim_thresh: R64,
        #[serde(default = "defaults::dets_for_track")]
        pub dets_for_track: usize,
        #[serde(default = "defaults::dets_for_show")]
        pub dets_for_show: usize,
        #[serde(default = "defaults::track_ciou_norm")]
        pub track_ciou_norm: R64,
        pub embedding_layer: Option<isize>,
        pub map: Option<PathBuf>,
        #[serde(with = "serde_anchors", default)]
        pub anchors: Option<Vec<(usize, usize)>>,
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }

    impl CommonLayerOptionsEx for Yolo {
        fn common(&self) -> &CommonLayerOptions {
            &self.common
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct BatchNorm {
        #[serde(flatten)]
        pub common: CommonLayerOptions,
    }

    impl CommonLayerOptionsEx for BatchNorm {
        fn common(&self) -> &CommonLayerOptions {
            &self.common
        }
    }

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
            step: usize,
            scale: R64,
        },
        Exp {
            gamma: R64,
        },
        Sigmoid {
            gamma: R64,
            step: usize,
        },
        Steps {
            steps: Vec<usize>,
            scales: Vec<R64>,
            seq_scales: Vec<R64>,
        },
        Sgdr,
        SgdrCustom {
            steps: Vec<usize>,
            scales: Vec<R64>,
            seq_scales: Vec<R64>,
        },
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize_repr, Deserialize_repr)]
    #[repr(usize)]
    pub enum MixUp {
        MixUp = 1,
        CutMix = 2,
        Mosaic = 3,
        Random = 4,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum Shape {
        Hwc([usize; 3]),
        Flat(usize),
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Adam {
        b1: R64,
        b2: R64,
        eps: R64,
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
        ReLU,
        #[serde(rename = "softmax")]
        Softmax,
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

    pub fn groups() -> usize {
        1
    }

    pub fn stride() -> usize {
        1
    }

    pub fn dilation() -> usize {
        1
    }

    pub fn angle() -> R64 {
        R64::new(15.0)
    }

    pub fn max_batches() -> usize {
        0
    }

    pub fn batch() -> usize {
        1
    }

    pub fn learning_rate() -> R64 {
        R64::new(0.001)
    }

    pub fn learning_rate_min() -> R64 {
        R64::new(0.00001)
    }

    pub fn sgdr_mult() -> usize {
        2
    }

    pub fn momentum() -> R64 {
        R64::new(0.9)
    }

    pub fn decay() -> R64 {
        R64::new(0.0001)
    }

    pub fn subdivisions() -> usize {
        1
    }

    pub fn time_steps() -> usize {
        1
    }

    pub fn track() -> usize {
        1
    }

    pub fn augment_speed() -> usize {
        2
    }

    pub fn loss_scale() -> R64 {
        R64::new(1.0)
    }

    pub fn workspace_size_limit_mb() -> usize {
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

    pub fn mixup() -> MixUp {
        MixUp::Random
    }

    pub fn label_smooth_eps() -> R64 {
        R64::new(0.0)
    }

    pub fn resize_step() -> usize {
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

    pub fn step() -> usize {
        1
    }

    pub fn scale() -> R64 {
        R64::new(1.0)
    }

    pub fn gamma() -> R64 {
        R64::new(1.0)
    }

    pub fn burn_in() -> usize {
        0
    }

    pub fn route_groups() -> usize {
        1
    }

    pub fn route_group_id() -> usize {
        0
    }

    pub fn weights_type() -> WeightsType {
        WeightsType::None
    }

    pub fn weights_normalization() -> WeightsNormalization {
        WeightsNormalization::None
    }

    pub fn maxpool_stride() -> usize {
        1
    }

    pub fn maxpool_depth() -> usize {
        0
    }

    pub fn out_channels() -> usize {
        1
    }

    pub fn upsample_stride() -> usize {
        2
    }

    pub fn upsample_scale() -> usize {
        1
    }

    pub fn classes() -> usize {
        warn!("classes option is not specified, use default 20");
        20
    }

    pub fn num() -> usize {
        1
    }

    pub fn max_boxes() -> usize {
        200
    }

    pub fn yolo_label_smooth_eps() -> R64 {
        R64::new(0.0)
    }

    pub fn scale_x_y() -> R64 {
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

    pub fn track_history_size() -> usize {
        5
    }

    pub fn sim_thresh() -> R64 {
        R64::new(0.8)
    }

    pub fn dets_for_track() -> usize {
        1
    }

    pub fn dets_for_show() -> usize {
        1
    }

    pub fn track_ciou_norm() -> R64 {
        R64::new(0.01)
    }

    pub fn connected_output() -> usize {
        1
    }

    pub fn connected_activation() -> Activation {
        Activation::Logistic
    }

    pub fn learning_scale_scale() -> R64 {
        R64::new(1.0)
    }
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

mod serde_vec_isize {
    use super::*;

    pub fn serialize<S>(steps: &Vec<isize>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        steps
            .iter()
            .map(|step| step.to_string())
            .join(",")
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<isize>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        let steps: Vec<isize> = text
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            .split(",")
            .map(|token| token.parse())
            .try_collect()
            .map_err(|err| D::Error::custom(format!("failed to parse steps: {:?}", err)))?;
        Ok(steps)
    }
}

mod serde_vec_usize {
    use super::*;

    pub fn serialize<S>(steps: &Vec<usize>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        steps
            .iter()
            .map(|step| step.to_string())
            .join(",")
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<usize>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        let steps: Vec<usize> = text
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            .split(",")
            .map(|token| token.parse())
            .try_collect()
            .map_err(|err| D::Error::custom(format!("failed to parse steps: {:?}", err)))?;
        Ok(steps)
    }
}

mod serde_opt_vec_usize {
    use super::*;

    pub fn serialize<S>(steps: &Option<Vec<usize>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        steps
            .as_ref()
            .map(|steps| steps.iter().map(|step| step.to_string()).join(","))
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Vec<usize>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = <Option<String>>::deserialize(deserializer)?;
        let steps: Option<Vec<usize>> = text
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

    pub fn serialize<S>(
        steps: &Option<Vec<(usize, usize)>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
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

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Vec<(usize, usize)>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = match Option::<String>::deserialize(deserializer)? {
            Some(text) => text,
            None => return Ok(None),
        };
        let values: Vec<usize> = text
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
