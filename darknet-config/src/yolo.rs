use super::{LayerIndex, Meta, OutputShape};
use crate::{common::*, utils};

#[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize, Hash)]
pub struct Yolo {
    pub classes: usize,
    #[serde(flatten)]
    pub anchors: Anchors,
    #[serde(rename = "max", default = "utils::integer::<_, 200>")]
    pub max_boxes: usize,
    pub max_delta: Option<R64>,
    #[serde(with = "utils::serde_comma_list", default)]
    pub counters_per_class: Option<Vec<usize>>,
    #[serde(default = "num_traits::zero")]
    pub label_smooth_eps: R64,
    #[serde(default = "num_traits::one")]
    pub scale_x_y: R64,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub objectness_smooth: bool,
    #[serde(default = "utils::ratio::<_, 75, 100>")]
    pub iou_normalizer: R64,
    #[serde(default = "num_traits::one")]
    pub obj_normalizer: R64,
    #[serde(default = "num_traits::one")]
    pub cls_normalizer: R64,
    #[serde(default = "num_traits::one")]
    pub delta_normalizer: R64,
    #[serde(default = "default_iou_loss")]
    pub iou_loss: IouLoss,
    #[serde(default = "default_iou_thresh_kind")]
    pub iou_thresh_kind: IouThreshold,
    #[serde(default = "num_traits::zero")]
    pub beta_nms: R64,
    #[serde(default = "default_nms_kind")]
    pub nms_kind: NmsKind,
    #[serde(default = "default_yolo_point")]
    pub yolo_point: YoloPoint,
    #[serde(default = "utils::ratio::<_, 2, 10>")]
    pub jitter: R64,
    #[serde(default = "num_traits::one")]
    pub resize: R64,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub focal_loss: bool,
    #[serde(default = "utils::ratio::<_, 5, 10>")]
    pub ignore_thresh: R64,
    #[serde(default = "num_traits::one")]
    pub truth_thresh: R64,
    #[serde(default = "num_traits::one")]
    pub iou_thresh: R64,
    #[serde(default = "num_traits::zero")]
    pub random: R64,
    #[serde(default = "utils::integer::<_, 5>")]
    pub track_history_size: usize,
    #[serde(default = "utils::ratio::<_, 8, 10>")]
    pub sim_thresh: R64,
    #[serde(default = "utils::integer::<_, 1>")]
    pub dets_for_track: usize,
    #[serde(default = "utils::integer::<_, 1>")]
    pub dets_for_show: usize,
    #[serde(default = "utils::ratio::<_, 1, 100>")]
    pub track_ciou_norm: R64,
    pub embedding_layer: Option<LayerIndex>,
    pub map: Option<PathBuf>,
    #[serde(flatten)]
    pub common: Meta,
}

impl Yolo {
    pub fn output_shape(&self, input_shape: [usize; 3]) -> Option<OutputShape> {
        let Self {
            classes,
            ref anchors,
            ..
        } = *self;
        let num_anchors = anchors.0.len();
        let [in_h, in_w, in_c] = input_shape;

        if in_c != num_anchors * (classes + 4 + 1) {
            return None;
        }

        Some(OutputShape::Yolo([in_h, in_w, in_c]))
    }
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

pub use anchors::*;
mod anchors {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct Anchor {
        pub enabled: bool,
        pub row: usize,
        pub col: usize,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
    #[serde(try_from = "RawAnchors", into = "RawAnchors")]
    pub struct Anchors(pub Vec<Anchor>);

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
    pub struct RawAnchors {
        // #[serde(default = "num_traits::one")]
        // pub num: usize,
        pub num: String,
        #[serde(with = "utils::serde_anchors", default)]
        pub anchors: Option<Vec<(usize, usize)>>,
        #[serde(with = "utils::serde_comma_list", default)]
        pub mask: Option<Vec<usize>>,
    }

    impl TryFrom<RawAnchors> for Anchors {
        type Error = Error;

        fn try_from(from: RawAnchors) -> Result<Self, Self::Error> {
            let RawAnchors { num, anchors, mask } = from;
            let num: usize = num
                .parse()
                .map_err(|_| anyhow!(r#"invalid "num" value {}"#, num))?;

            let anchors_vec: Vec<_> = match (num, anchors, mask) {
                (0, None, None) => vec![],
                (num, Some(anchors), mask) => {
                    let num_anchors = anchors.len();

                    match num.cmp(&num_anchors) {
                        Greater => {
                            bail!(
                                r#"num={} is greater than number of anchors ({})"#,
                                num,
                                num_anchors
                            )
                        }
                        Equal => {}
                        Less => {
                            warn!(
                                r#"num={} is less than number of anchors ({})"#,
                                num, num_anchors
                            );
                        }
                    }

                    let mask_set: HashSet<_> = mask
                        .iter()
                        .flatten()
                        .cloned()
                        .map(|index: usize| -> Result<_> {
                            ensure!(
                                index < anchors.len(),
                                "mask index {} exceeds the length of anchors ({})",
                                index,
                                anchors.len()
                            );
                            Ok(index)
                        })
                        .try_collect()?;

                    let anchors_vec: Vec<_> = anchors[0..num]
                        .iter()
                        .cloned()
                        .enumerate()
                        .map(|(index, (row, col))| Anchor {
                            enabled: mask_set.contains(&index),
                            row,
                            col,
                        })
                        .collect();

                    anchors_vec
                }
                _ => {
                    bail!(r#"the "num", length of "anchors" and indexes of "mask" does not match"#)
                }
            };

            Ok(Self(anchors_vec))
        }
    }

    impl From<Anchors> for RawAnchors {
        fn from(from: Anchors) -> Self {
            let mask: Vec<_> = from
                .0
                .iter()
                .enumerate()
                .filter_map(|(index, anchor)| anchor.enabled.then(|| index))
                .collect();

            let anchors: Vec<(usize, usize)> = from
                .0
                .into_iter()
                .map(|anchor| {
                    let Anchor { row, col, .. } = anchor;
                    (row, col)
                })
                .collect();

            Self {
                num: anchors.len().to_string(),
                anchors: (!anchors.is_empty()).then(|| anchors),
                mask: (!mask.is_empty()).then(|| mask),
            }
        }
    }
}

fn default_iou_loss() -> IouLoss {
    IouLoss::Mse
}

fn default_iou_thresh_kind() -> IouThreshold {
    IouThreshold::IoU
}

fn default_nms_kind() -> NmsKind {
    NmsKind::Default
}

fn default_yolo_point() -> YoloPoint {
    YoloPoint::Center
}
