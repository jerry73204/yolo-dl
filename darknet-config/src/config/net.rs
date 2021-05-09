use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "RawNet", into = "RawNet")]
pub struct Net {
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
}

impl Net {
    pub fn iteration(&self, seen: u64) -> u64 {
        seen / (self.batch * self.subdivisions)
    }
}

impl TryFrom<RawNet> for Net {
    type Error = Error;

    fn try_from(from: RawNet) -> Result<Self, Self::Error> {
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
        } = from;

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
pub(super) struct RawNet {
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
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub try_fix_nan: bool,
    #[serde(default = "defaults::loss_scale")]
    pub loss_scale: R64,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub dynamic_minibatch: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub optimized_memory: bool,
    #[serde(
        rename = "workspace_size_limit_MB",
        default = "defaults::workspace_size_limit_mb"
    )]
    pub workspace_size_limit_mb: u64,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
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
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_true")]
    pub flip: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub blur: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub gaussian_noise: bool,
    #[serde(default = "defaults::mixup")]
    pub mixup: MixUp,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub cutmux: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub mosaic: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub letter_box: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub mosaic_bound: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub contrastive: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub contrastive_jit_flip: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub contrastive_color: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub unsupervised: bool,
    #[serde(default = "defaults::label_smooth_eps")]
    pub label_smooth_eps: R64,
    #[serde(default = "defaults::resize_step")]
    pub resize_step: u64,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
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
    #[serde(with = "serde_::net_steps", default)]
    pub steps: Option<Vec<u64>>,
    #[serde(with = "serde_::opt_vec_r64", default)]
    pub scales: Option<Vec<R64>>,
    #[serde(with = "serde_::opt_vec_r64", default)]
    pub seq_scales: Option<Vec<R64>>,
    #[serde(default = "defaults::gamma")]
    pub gamma: R64,
}

impl From<Net> for RawNet {
    fn from(from: Net) -> Self {
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
        } = from;

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
        }
    }
}
