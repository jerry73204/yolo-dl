use super::{Adam, MixUp, Policy, PolicyKind, Shape};
use crate::{common::*, utils, utils::default};

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

impl Net {
    pub fn iteration(&self, seen: usize) -> usize {
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
            Some(Adam {
                b1: b1,
                b2: b2,
                eps: eps,
            })
        } else {
            None
        };
        let max_crop = max_crop.unwrap_or_else(|| width.map(|w| w.get()).unwrap_or(0) * 2);
        let min_crop = min_crop.unwrap_or_else(|| width.map(|w| w.get()).unwrap_or(0));
        let input_size = match (inputs, height, width, channels) {
            (Some(inputs), None, None, None) => Shape::Dim1(inputs.get()),
            (None, Some(height), Some(width), Some(channels)) => {
                Shape::Dim3([height.get(), width.get(), channels.get()])
            }
            _ => bail!("either inputs or height/width/channels must be specified"),
        };
        let policy = match policy {
            PolicyKind::Random => Policy::Random,
            PolicyKind::Poly => Policy::Poly,
            PolicyKind::Constant => Policy::Constant,
            PolicyKind::Step => Policy::Step {
                step: step,
                scale: scale,
            },
            PolicyKind::Exp => Policy::Exp { gamma: gamma },
            PolicyKind::Sigmoid => Policy::Sigmoid {
                gamma: gamma,
                step: step,
            },
            PolicyKind::Steps => {
                let steps =
                    steps.ok_or_else(|| anyhow!("steps must be specified for step policy"))?;
                let scales = {
                    let scales = scales
                        .ok_or_else(|| anyhow!("scales must be specified for step policy"))?;
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
    #[serde(default = "utils::integer::<_, 0>")]
    pub max_batches: usize,
    #[serde(default = "utils::integer::<_, 1>")]
    pub batch: usize,
    #[serde(default = "utils::ratio::<_, 1, 1000>")]
    pub learning_rate: R64,
    #[serde(default = "utils::ratio::<_, 1, 100000>")]
    pub learning_rate_min: R64,
    pub sgdr_cycle: Option<usize>,
    #[serde(default = "utils::integer::<_, 2>")]
    pub sgdr_mult: usize,
    #[serde(default = "utils::ratio::<_, 9, 10>")]
    pub momentum: R64,
    #[serde(default = "utils::ratio::<_, 1, 10000>")]
    pub decay: R64,
    #[serde(default = "utils::integer::<_, 1>")]
    pub subdivisions: usize,
    #[serde(default = "utils::integer::<_, 1>")]
    pub time_steps: usize,
    #[serde(default = "utils::integer::<_, 1>")]
    pub track: usize,
    #[serde(default = "utils::integer::<_, 2>")]
    pub augment_speed: usize,
    pub sequential_subdivisions: Option<usize>,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub try_fix_nan: bool,
    #[serde(default = "utils::ratio::<_, 1, 1>")]
    pub loss_scale: R64,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub dynamic_minibatch: bool,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub optimized_memory: bool,
    #[serde(
        rename = "workspace_size_limit_MB",
        default = "utils::integer::<_, 1024>"
    )]
    pub workspace_size_limit_mb: usize,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub adam: bool,
    #[serde(default = "utils::ratio::<_, 9, 10>")]
    pub b1: R64,
    #[serde(default = "utils::ratio::<_, 999, 1000>")]
    pub b2: R64,
    #[serde(default = "utils::ratio::<_, 1, 1000000>")]
    pub eps: R64,
    pub width: Option<NonZeroUsize>,
    pub height: Option<NonZeroUsize>,
    pub channels: Option<NonZeroUsize>,
    pub inputs: Option<NonZeroUsize>,
    pub max_crop: Option<usize>,
    pub min_crop: Option<usize>,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_true")]
    pub flip: bool,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub blur: bool,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub gaussian_noise: bool,
    #[serde(default = "default_mixup")]
    pub mixup: MixUp,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub cutmux: bool,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub mosaic: bool,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub letter_box: bool,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub mosaic_bound: bool,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub contrastive: bool,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub contrastive_jit_flip: bool,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub contrastive_color: bool,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub unsupervised: bool,
    #[serde(default = "utils::ratio::<_, 0, 1>")]
    pub label_smooth_eps: R64,
    #[serde(default = "utils::integer::<_, 32>")]
    pub resize_step: usize,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub attention: bool,
    #[serde(default = "utils::ratio::<_, 0, 1>")]
    pub adversarial_lr: R64,
    #[serde(default = "utils::ratio::<_, 20, 1>")]
    pub max_chart_loss: R64,
    #[serde(default = "utils::ratio::<_, 15, 1>")]
    pub angle: R64,
    #[serde(default = "utils::ratio::<_, 1, 1>")]
    pub aspect: R64,
    #[serde(default = "utils::ratio::<_, 1, 1>")]
    pub saturation: R64,
    #[serde(default = "utils::ratio::<_, 1, 1>")]
    pub exposure: R64,
    #[serde(default = "utils::ratio::<_, 0, 1>")]
    pub hue: R64,
    #[serde(default = "utils::ratio::<_, 4, 1>")]
    pub power: R64,
    #[serde(default = "default_policy")]
    pub policy: PolicyKind,
    #[serde(default = "utils::integer::<_, 0>")]
    pub burn_in: usize,
    #[serde(default = "utils::integer::<_, 1>")]
    pub step: usize,
    #[serde(default = "utils::ratio::<_, 1, 1>")]
    pub scale: R64,
    #[serde(with = "serde_net_steps", default)]
    pub steps: Option<Vec<usize>>,
    #[serde(with = "utils::serde_r64_comma_list", default)]
    pub scales: Option<Vec<R64>>,
    #[serde(with = "utils::serde_r64_comma_list", default)]
    pub seq_scales: Option<Vec<R64>>,
    #[serde(default = "utils::ratio::<_, 1, 1>")]
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
            None => (false, default(), default(), default()),
        };

        let (inputs, height, width, channels) = match input_size {
            Shape::Dim3([height, width, channels]) => {
                (None, Some(height), Some(width), Some(channels))
            }
            Shape::Dim1(inputs) => (Some(inputs), None, None, None),
        };

        let (policy, step, scale, steps, scales, seq_scales, gamma) = match policy {
            Policy::Random => (
                PolicyKind::Random,
                default(),
                default(),
                None,
                None,
                None,
                default(),
            ),
            Policy::Poly => (
                PolicyKind::Poly,
                default(),
                default(),
                None,
                None,
                None,
                default(),
            ),
            Policy::Constant => (
                PolicyKind::Constant,
                default(),
                default(),
                None,
                None,
                None,
                default(),
            ),
            Policy::Step { step, scale } => {
                (PolicyKind::Step, step, scale, None, None, None, default())
            }
            Policy::Exp { gamma } => (
                PolicyKind::Exp,
                default(),
                default(),
                None,
                None,
                None,
                gamma,
            ),
            Policy::Sigmoid { gamma, step } => (
                PolicyKind::Sigmoid,
                step,
                default(),
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
                default(),
                default(),
                Some(steps),
                Some(scales),
                Some(seq_scales),
                default(),
            ),
            Policy::Sgdr => (
                PolicyKind::Sgdr,
                default(),
                default(),
                None,
                None,
                None,
                default(),
            ),
            Policy::SgdrCustom {
                steps,
                scales,
                seq_scales,
            } => (
                PolicyKind::Sgdr,
                default(),
                default(),
                Some(steps),
                Some(scales),
                Some(seq_scales),
                default(),
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
            b1: b1.into(),
            b2: b2.into(),
            eps: eps.into(),
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
            step: step.into(),
            scale: scale.into(),
            steps,
            scales,
            seq_scales,
            gamma: gamma.into(),
        }
    }
}

fn default_mixup() -> MixUp {
    MixUp::Random
}

fn default_policy() -> PolicyKind {
    PolicyKind::Constant
}

mod serde_net_steps {
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
                .map(|text| -> Result<_, String> {
                    let steps: Vec<usize> = text.split(',')
                        .enumerate()
                        .map(|(index, token)| {
                            let step: isize  = token
                                .trim()
                                .parse()
                                .map_err(|_| format!("'{}' is not an integer", token))?;

                            let step: usize = match (index, step) {
                                (0, -1) => {
                                    warn!("the first -1 in 'steps' option is regarded as 0");
                                    0
                                }
                                (0, step) => {
                                    if step < 0 {
                                        return Err(format!("invalid steps '{}': the first step must be -1 or non-negative integer", text));
                                    }
                                    step as usize
                                }
                                (_, step) => {
                                    if step < 0 {
                                        return Err(format!("invalid steps '{}': all steps except the first step must be positive integer", text));
                                    }
                                    step as usize
                                }
                            };

                            Ok(step)
                        })
                        .try_collect()?;

                    let is_monotonic = steps.iter().scan(None, |prev, curr| {
                        match prev {
                            None => None,
                            Some(None) => {*prev = Some(Some(curr)); Some(true)}
                            Some(Some(prev_val)) => {
                                if *prev_val < curr {
                                    *prev = Some(Some(curr));
                                    Some(true)
                                } else {
                                    *prev = None;
                                    Some(false)
                                }
                            }
                        }
                    })
                    .all(|yes| yes);

                    if !is_monotonic {
                        return Err(format!("the steps '{}' is not monotonic", text));
                    }

                    Ok(steps)
                })
                .transpose()
                .map_err(|err| D::Error::custom(format!("failed to parse steps: {:?}", err)))?;
        Ok(steps)
    }
}
