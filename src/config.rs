use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "Vec<Item>")]
#[serde(into = "Vec<Item>")]
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
                    Item::Convolutional(layer) => Layer::Convolutional(layer),
                    Item::Route(layer) => Layer::Route(layer),
                    Item::Shortcut(layer) => Layer::Shortcut(layer),
                    Item::MaxPool(layer) => Layer::MaxPool(layer),
                    Item::UpSample(layer) => Layer::UpSample(layer),
                    Item::Yolo(layer) => Layer::Yolo(layer),
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
}

impl From<Config> for Vec<Item> {
    fn from(config: Config) -> Self {
        let Config { net, layers } = config;
        let items: Vec<_> = iter::once(Item::Net(net))
            .chain(layers.into_iter().map(|layer| match layer {
                Layer::Convolutional(layer) => Item::Convolutional(layer),
                Layer::Route(layer) => Item::Route(layer),
                Layer::Shortcut(layer) => Item::Shortcut(layer),
                Layer::MaxPool(layer) => Item::MaxPool(layer),
                Layer::UpSample(layer) => Item::UpSample(layer),
                Layer::Yolo(layer) => Item::Yolo(layer),
            }))
            .collect();
        items
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Item {
    #[serde(rename = "net")]
    Net(Net),
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
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "RawNet")]
#[serde(into = "RawNet")]
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
    pub workspace_size_limit_MB: usize,
    pub adam: Option<Adam>,
    pub input_size: InputSize,
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
            workspace_size_limit_MB,
            adam,
            B1,
            B2,
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
            Some(Adam {
                b1: B1,
                b2: B2,
                eps,
            })
        } else {
            None
        };
        let max_crop = max_crop.unwrap_or_else(|| width.map(|w| w.get()).unwrap_or(0) * 2);
        let min_crop = min_crop.unwrap_or_else(|| width.map(|w| w.get()).unwrap_or(0));
        let input_size = match (inputs, height, width, channels) {
            (Some(inputs), None, None, None) => InputSize::Inputs {
                inputs: inputs.get(),
            },
            (None, Some(height), Some(width), Some(channels)) => InputSize::Hwc {
                height: height.get(),
                width: width.get(),
                channels: channels.get(),
            },
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
                let steps =
                    steps.ok_or_else(|| format_err!("steps must be specified for step policy"))?;
                let scales = {
                    let scales = scales
                        .ok_or_else(|| format_err!("scales must be specified for step policy"))?;
                    ensure!(
                        steps.len() == scales.len(),
                        "the length of steps and scales must be equal"
                    );
                    scales
                };
                let seq_scales = {
                    let seq_scales = seq_scales.unwrap_or_else(|| vec![R64::new(1.0); steps.len()]);
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
                    let seq_scales = seq_scales.unwrap_or_else(|| vec![R64::new(1.0); steps.len()]);
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
            workspace_size_limit_MB,
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
    #[serde(default = "default_max_batches")]
    pub max_batches: usize,
    #[serde(default = "default_batch")]
    pub batch: usize,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: R64,
    #[serde(default = "default_learning_rate_min")]
    pub learning_rate_min: R64,
    pub sgdr_cycle: Option<usize>,
    #[serde(default = "default_sgdr_mult")]
    pub sgdr_mult: usize,
    #[serde(default = "default_momentum")]
    pub momentum: R64,
    #[serde(default = "default_decay")]
    pub decay: R64,
    #[serde(default = "default_subdivisions")]
    pub subdivisions: usize,
    #[serde(default = "default_time_steps")]
    pub time_steps: usize,
    #[serde(default = "default_track")]
    pub track: usize,
    #[serde(default = "default_augment_speed")]
    pub augment_speed: usize,
    pub sequential_subdivisions: Option<usize>,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub try_fix_nan: bool,
    #[serde(default = "default_loss_scale")]
    pub loss_scale: R64,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub dynamic_minibatch: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub optimized_memory: bool,
    #[serde(default = "default_workspace_size_limit_MB")]
    pub workspace_size_limit_MB: usize,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub adam: bool,
    #[serde(default = "default_B1")]
    pub B1: R64,
    #[serde(default = "default_B2")]
    pub B2: R64,
    #[serde(default = "default_eps")]
    pub eps: R64,
    pub width: Option<NonZeroUsize>,
    pub height: Option<NonZeroUsize>,
    pub channels: Option<NonZeroUsize>,
    pub inputs: Option<NonZeroUsize>,
    pub max_crop: Option<usize>,
    pub min_crop: Option<usize>,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_true")]
    pub flip: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub blur: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub gaussian_noise: bool,
    #[serde(default = "default_mixup")]
    pub mixup: MixUp,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub cutmux: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub mosaic: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub letter_box: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub mosaic_bound: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub contrastive: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub contrastive_jit_flip: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub contrastive_color: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub unsupervised: bool,
    #[serde(default = "default_label_smooth_eps")]
    pub label_smooth_eps: R64,
    #[serde(default = "default_resize_step")]
    pub resize_step: usize,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub attention: bool,
    #[serde(default = "default_adversarial_lr")]
    pub adversarial_lr: R64,
    #[serde(default = "default_max_chart_loss")]
    pub max_chart_loss: R64,
    #[serde(default = "default_angle")]
    pub angle: R64,
    #[serde(default = "default_aspect")]
    pub aspect: R64,
    #[serde(default = "default_saturation")]
    pub saturation: R64,
    #[serde(default = "default_exposure")]
    pub exposure: R64,
    #[serde(default = "default_hue")]
    pub hue: R64,
    #[serde(default = "default_power")]
    pub power: R64,
    #[serde(default = "default_policy")]
    pub policy: PolicyKind,
    #[serde(default = "default_burn_in")]
    pub burn_in: usize,
    #[serde(default = "default_step")]
    pub step: usize,
    #[serde(default = "default_scale")]
    pub scale: R64,
    #[serde(with = "serde_opt_vec_usize", default)]
    pub steps: Option<Vec<usize>>,
    #[serde(with = "serde_opt_vec_r64", default)]
    pub scales: Option<Vec<R64>>,
    #[serde(with = "serde_opt_vec_r64", default)]
    pub seq_scales: Option<Vec<R64>>,
    #[serde(default = "default_gamma")]
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
            workspace_size_limit_MB,
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

        let (adam, B1, B2, eps) = match adam {
            Some(Adam { b1, b2, eps }) => (true, b1, b2, eps),
            None => (false, default_B1(), default_B2(), default_eps()),
        };
        let (inputs, height, width, channels) = match input_size {
            InputSize::Hwc {
                height,
                width,
                channels,
            } => (None, Some(height), Some(width), Some(channels)),
            InputSize::Inputs { inputs } => (Some(inputs), None, None, None),
        };

        let (policy, step, scale, steps, scales, seq_scales, gamma) = match policy {
            Policy::Random => (
                PolicyKind::Random,
                default_step(),
                default_scale(),
                None,
                None,
                None,
                default_gamma(),
            ),
            Policy::Poly => (
                PolicyKind::Poly,
                default_step(),
                default_scale(),
                None,
                None,
                None,
                default_gamma(),
            ),
            Policy::Constant => (
                PolicyKind::Constant,
                default_step(),
                default_scale(),
                None,
                None,
                None,
                default_gamma(),
            ),
            Policy::Step { step, scale } => (
                PolicyKind::Step,
                step,
                scale,
                None,
                None,
                None,
                default_gamma(),
            ),
            Policy::Exp { gamma } => (
                PolicyKind::Exp,
                default_step(),
                default_scale(),
                None,
                None,
                None,
                gamma,
            ),
            Policy::Sigmoid { gamma, step } => (
                PolicyKind::Sigmoid,
                step,
                default_scale(),
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
                default_step(),
                default_scale(),
                Some(steps),
                Some(scales),
                Some(seq_scales),
                default_gamma(),
            ),
            Policy::Sgdr => (
                PolicyKind::Sgdr,
                default_step(),
                default_scale(),
                None,
                None,
                None,
                default_gamma(),
            ),
            Policy::SgdrCustom {
                steps,
                scales,
                seq_scales,
            } => (
                PolicyKind::Sgdr,
                default_step(),
                default_scale(),
                Some(steps),
                Some(scales),
                Some(seq_scales),
                default_gamma(),
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
            workspace_size_limit_MB,
            adam,
            B1,
            B2,
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
#[serde(try_from = "RawConvolutional")]
#[serde(into = "RawConvolutional")]
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
            bin_output,
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
        } = raw;

        let stride_x = stride_x.unwrap_or(stride);
        let stride_y = stride_y.unwrap_or(stride);

        ensure!(
            size != 1 || dilation == 1,
            "dilation must be 1 if size is 1"
        );

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

        match (deform, size == 1) {
            (Deform::None, _) => (),
            (_, false) => (),
            (_, true) => bail!("sway, rotate, stretch, stretch_sway shoud be used with size >= 3"),
        };

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
            use_bin_output: bin_output,
            deform,
            flipped,
            dot,
            angle,
            grad_centr,
            reverse,
            coordconv,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RawConvolutional {
    pub filters: usize,
    #[serde(default = "default_groups")]
    pub groups: usize,
    pub size: usize,
    #[serde(default = "default_stride")]
    pub stride: usize,
    pub stride_x: Option<usize>,
    pub stride_y: Option<usize>,
    #[serde(default = "default_dilation")]
    pub dilation: usize,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub antialiasing: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub pad: bool,
    pub padding: Option<usize>,
    pub activation: Activation,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub assisted_excitation: bool,
    pub share_index: Option<usize>,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub batch_normalize: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub cbn: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub binary: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub xnor: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub bin_output: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub sway: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub rotate: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub stretch: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub stretch_sway: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub flipped: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub dot: bool,
    #[serde(default = "default_angle")]
    pub angle: R64,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub grad_centr: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub reverse: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub coordconv: bool,
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
            use_bin_output: bin_output,
            deform,
            flipped,
            dot,
            angle,
            grad_centr,
            reverse,
            coordconv,
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
            stride: default_stride(),
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
            bin_output,
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
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Route {
    #[serde(with = "serde_vec_isize")]
    pub layers: Vec<isize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shortcut {
    pub from: isize,
    pub activation: Activation,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MaxPool {
    pub stride: usize,
    pub size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UpSample {
    pub stride: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Yolo {
    #[serde(with = "serde_vec_usize")]
    pub mask: Vec<usize>,
    #[serde(with = "serde_anchors")]
    pub anchors: Vec<(usize, usize)>,
    pub classes: usize,
    pub num: usize,
    pub jitter: R64,
    pub ignore_thresh: R64,
    pub truth_thresh: R64,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub random: bool,
    pub scale_x_y: R64,
    pub iou_thresh: R64,
    pub cls_normalizer: R64,
    pub iou_normalizer: R64,
    pub iou_loss: IouLoss,
    pub nms_kind: NmsKind,
    pub beta_nms: R64,
    pub max_delta: R64,
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
pub enum InputSize {
    Hwc {
        height: usize,
        width: usize,
        channels: usize,
    },
    Inputs {
        inputs: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Adam {
    b1: R64,
    b2: R64,
    eps: R64,
}

// utility functions

fn default_bool_true() -> bool {
    true
}

fn default_bool_false() -> bool {
    false
}

fn default_groups() -> usize {
    1
}

fn default_stride() -> usize {
    1
}

fn default_dilation() -> usize {
    1
}

fn default_angle() -> R64 {
    R64::new(15.0)
}

fn default_max_batches() -> usize {
    0
}

fn default_batch() -> usize {
    1
}

fn default_learning_rate() -> R64 {
    R64::new(0.001)
}

fn default_learning_rate_min() -> R64 {
    R64::new(0.00001)
}

fn default_sgdr_mult() -> usize {
    2
}

fn default_momentum() -> R64 {
    R64::new(0.9)
}

fn default_decay() -> R64 {
    R64::new(0.0001)
}

fn default_subdivisions() -> usize {
    1
}

fn default_time_steps() -> usize {
    1
}

fn default_track() -> usize {
    1
}

fn default_augment_speed() -> usize {
    2
}

fn default_loss_scale() -> R64 {
    R64::new(1.0)
}

fn default_workspace_size_limit_MB() -> usize {
    1024
}

fn default_B1() -> R64 {
    R64::new(0.9)
}

fn default_B2() -> R64 {
    R64::new(0.999)
}

fn default_eps() -> R64 {
    R64::new(0.000001)
}

fn default_mixup() -> MixUp {
    MixUp::Random
}

fn default_label_smooth_eps() -> R64 {
    R64::new(0.0)
}

fn default_resize_step() -> usize {
    32
}

fn default_adversarial_lr() -> R64 {
    R64::new(0.0)
}

fn default_max_chart_loss() -> R64 {
    R64::new(20.0)
}

fn default_aspect() -> R64 {
    R64::new(1.0)
}

fn default_saturation() -> R64 {
    R64::new(1.0)
}

fn default_exposure() -> R64 {
    R64::new(1.0)
}

fn default_hue() -> R64 {
    R64::new(0.0)
}

fn default_power() -> R64 {
    R64::new(4.0)
}

fn default_policy() -> PolicyKind {
    PolicyKind::Constant
}

fn default_step() -> usize {
    1
}

fn default_scale() -> R64 {
    R64::new(1.0)
}

fn default_gamma() -> R64 {
    R64::new(1.0)
}

fn default_burn_in() -> usize {
    0
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

    pub fn serialize<S>(steps: &Vec<(usize, usize)>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        steps
            .iter()
            .flat_map(|(w, h)| vec![w, h])
            .map(|val| val.to_string())
            .join(",")
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<(usize, usize)>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
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
        Ok(anchors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wtf() -> Result<()> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("yolov4.cfg");
        let config: Config = serde_ini::from_str(&fs::read_to_string(path)?)?;
        dbg!(config);
        Ok(())
    }
}
