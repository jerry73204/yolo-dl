use crate::common::*;

mod avg_pool;
mod batch_norm;
mod common;
mod connected;
mod convolutional;
mod cost;
mod crop;
mod darknet;
mod dropout;
mod gaussian_yolo;
mod item;
mod max_pool;
mod misc;
mod net;
mod route;
mod shortcut;
mod softmax;
mod unimplemented;
mod up_sample;
mod yolo;

pub use self::darknet::*;
pub use avg_pool::*;
pub use batch_norm::*;
pub use common::*;
pub use connected::*;
pub use convolutional::*;
pub use cost::*;
pub use crop::*;
pub use dropout::*;
pub use gaussian_yolo::*;
use item::*;
pub use max_pool::*;
pub use misc::*;
pub use net::*;
pub use route::*;
pub use shortcut::*;
pub use softmax::*;
pub use unimplemented::*;
pub use up_sample::*;
pub use yolo::*;

// utility functions

mod defaults {
    use super::*;

    pub fn bool_true() -> bool {
        true
    }

    pub fn bool_false() -> bool {
        false
    }

    pub fn stop_backward() -> u64 {
        0
    }

    pub fn crop_stride() -> u64 {
        1
    }

    pub fn cost_type() -> CostType {
        CostType::Sse
    }

    pub fn cost_scale() -> R64 {
        r64(1.0)
    }

    pub fn cost_ratio() -> R64 {
        r64(0.0)
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

    pub fn learning_rate_scale() -> R64 {
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

fn hash_vec<T, H>(layers: &IndexSet<T>, state: &mut H)
where
    T: Hash,
    H: Hasher,
{
    let layers: Vec<_> = layers.iter().collect();
    layers.hash(state);
}

// fn hash_vec_indexset<T, H>(set: &IndexSet<T>, state: &mut H)
// where
//     T: Hash,
//     H: Hasher,
// {
//     let set: Vec<_> = set.iter().collect();
//     set.hash(state);
// }

fn hash_option_vec_indexset<T, H>(opt: &Option<IndexSet<T>>, state: &mut H)
where
    T: Hash,
    H: Hasher,
{
    let opt: Option<Vec<_>> = opt.as_ref().map(|set| set.iter().collect());
    opt.hash(state);
}

mod serde_ {
    use super::*;

    pub mod zero_one_bool {
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

    pub mod vec_layers {
        use super::*;

        pub fn serialize<S>(
            indexes: &IndexSet<LayerIndex>,
            serializer: S,
        ) -> Result<S::Ok, S::Error>
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
                .split(',')
                .map(|token| -> Result<_, String> {
                    let index: isize = token
                        .trim()
                        .parse()
                        .map_err(|_| format!("{} is not a valid index", token))?;
                    let index = LayerIndex::from(index);
                    Ok(index)
                })
                .try_collect()
                .map_err(|err| {
                    D::Error::custom(format!("failed to parse layer index: {:?}", err))
                })?;
            let layers_set: IndexSet<LayerIndex> = layers_vec.iter().cloned().collect();

            if layers_vec.len() != layers_set.len() {
                return Err(D::Error::custom("duplicated layer index is not allowed"));
            }

            Ok(layers_set)
        }
    }

    pub mod mask {
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
                .split(',')
                .map(|token| {
                    token
                        .trim()
                        .parse()
                        .map_err(|_| format!("'{}' is not a valid index", token))
                })
                .try_collect()
                .map_err(|err| D::Error::custom(format!("failed to parse steps: {:?}", err)))?;
            let steps_set: IndexSet<_> = steps_vec.iter().cloned().collect();

            if steps_vec.len() != steps_set.len() {
                return Err(D::Error::custom("duplicated mask indexes is not allowed"));
            }

            Ok(Some(steps_set))
        }
    }

    pub mod net_steps {
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
                .map(|text| -> Result<_, String> {
                    let steps: Vec<u64> = text.split(',')
                        .enumerate()
                        .map(|(index, token)| {
                            let step:i64  = token
                                .trim()
                                .parse()
                                .map_err(|_| format!("'{}' is not an integer", token))?;

                            let step: u64 = match (index, step) {
                                (0, -1) => {
                                    warn!("the first -1 in 'steps' option is regarded as 0");
                                    0
                                }
                                (0, step) => {
                                    if step < 0 {
                                        return Err(format!("invalid steps '{}': the first step must be -1 or non-negative integer", text));
                                    }
                                    step as u64
                                }
                                (_, step) => {
                                    if step < 0 {
                                        return Err(format!("invalid steps '{}': all steps except the first step must be positive integer", text));
                                    }
                                    step as u64
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

    pub mod opt_vec_u64 {
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
                    text.split(',')
                        .map(|token| {
                            token
                                .trim()
                                .parse()
                                .map_err(|_| format!("'{}' is not a non-negative integer", token))
                        })
                        .try_collect()
                })
                .transpose()
                .map_err(|err| D::Error::custom(format!("failed to parse steps: {:?}", err)))?;
            Ok(steps)
        }
    }

    pub mod opt_vec_r64 {
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
                        .split(',')
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

    pub mod anchors {
        use super::*;

        pub fn serialize<S>(
            steps: &Option<Vec<(u64, u64)>>,
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

        pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Vec<(u64, u64)>>, D::Error>
        where
            D: Deserializer<'de>,
        {
            let text = match Option::<String>::deserialize(deserializer)? {
                Some(text) => text,
                None => return Ok(None),
            };
            let values: Vec<u64> = text
                .split(',')
                .map(|token| {
                    token
                        .trim()
                        .parse()
                        .map_err(|_| format!("{} is not a number", token))
                })
                .try_collect()
                .map_err(|err| D::Error::custom(format!("failed to parse anchors: {:?}", err)))?;

            if values.len() % 2 != 0 {
                return Err(D::Error::custom(format!(
                    "expect even number of values in '{}'",
                    text
                )));
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

    pub mod weights_type {
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
}
