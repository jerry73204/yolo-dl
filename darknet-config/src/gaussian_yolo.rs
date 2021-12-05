use super::{IouLoss, IouThreshold, Meta, NmsKind, OutputShape, YoloPoint};
use crate::{common::*, utils};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Derivative, Serialize, Deserialize)]
#[serde(try_from = "RawGaussianYolo")]
pub struct GaussianYolo {
    pub classes: usize,
    pub max_boxes: usize,
    pub max_delta: Option<R64>,
    pub counters_per_class: Option<Vec<usize>>,
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
    pub ignore_thresh: R64,
    pub truth_thresh: R64,
    pub iou_thresh: R64,
    pub random: R64,
    pub map: Option<PathBuf>,
    pub anchors: Vec<(usize, usize)>,
    pub yolo_point: YoloPoint,
    pub iou_loss: IouLoss,
    pub nms_kind: NmsKind,
    pub common: Meta,
}

impl GaussianYolo {
    pub fn output_shape(&self, input_shape: [usize; 3]) -> Option<OutputShape> {
        let Self {
            classes,
            ref anchors,
            ..
        } = *self;
        let num_anchors = anchors.len();
        let [in_h, in_w, in_c] = input_shape;

        if in_c != num_anchors * (classes + 4 + 1) {
            return None;
        }

        Some(OutputShape::Yolo([in_h, in_w, in_c]))
    }
}

impl TryFrom<RawGaussianYolo> for GaussianYolo {
    type Error = Error;

    fn try_from(from: RawGaussianYolo) -> Result<Self, Self::Error> {
        let RawGaussianYolo {
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
        } = from;

        let mask = mask.unwrap_or_else(Vec::new);
        let anchors = match (num, anchors) {
            (0, None) => vec![],
            (_, None) => bail!("num and length of anchors mismatch"),
            (_, Some(anchors)) => {
                ensure!(
                    anchors.len() == num as usize,
                    "num and length of anchors mismatch"
                );
                let anchors: Option<Vec<_>> = mask
                    .into_iter()
                    .map(|index| anchors.get(index as usize).copied())
                    .collect();
                anchors.ok_or_else(|| anyhow!("mask index exceeds total number of anchors"))?
            }
        };

        Ok(GaussianYolo {
            classes,
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
            iou_thresh_kind,
            beta_nms,
            jitter,
            resize,
            ignore_thresh,
            truth_thresh,
            iou_thresh,
            random,
            map,
            anchors,
            yolo_point,
            iou_loss,
            nms_kind,
            common,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
#[derivative(Hash)]
pub(super) struct RawGaussianYolo {
    pub classes: usize,
    #[serde(rename = "max", default = "utils::integer::<_, 200>")]
    pub max_boxes: usize,
    #[serde(default = "num_traits::one")]
    pub num: usize,
    #[serde(with = "utils::serde_comma_list", default)]
    pub mask: Option<Vec<usize>>,
    pub max_delta: Option<R64>,
    #[serde(with = "utils::serde_comma_list", default)]
    pub counters_per_class: Option<Vec<usize>>,
    #[serde(default = "num_traits::zero")]
    pub label_smooth_eps: R64,
    #[serde(default = "num_traits::one")]
    pub scale_x_y: R64,
    #[serde(with = "utils::zero_one_bool", default = "utils::bool_false")]
    pub objectness_smooth: bool,
    #[serde(default = "num_traits::one")]
    pub uc_normalizer: R64,
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
    #[serde(default = "utils::ratio::<_, 5, 10>")]
    pub ignore_thresh: R64,
    #[serde(default = "num_traits::one")]
    pub truth_thresh: R64,
    #[serde(default = "num_traits::one")]
    pub iou_thresh: R64,
    #[serde(default = "num_traits::zero")]
    pub random: R64,
    pub map: Option<PathBuf>,
    #[serde(with = "utils::serde_anchors", default)]
    pub anchors: Option<Vec<(usize, usize)>>,
    #[serde(flatten)]
    pub common: Meta,
}

// pub fn classes() -> usize {
//     warn!("classes option is not specified, use default 20");
//     20
// }

pub fn default_iou_loss() -> IouLoss {
    IouLoss::Mse
}

pub fn default_iou_thresh_kind() -> IouThreshold {
    IouThreshold::IoU
}

pub fn default_nms_kind() -> NmsKind {
    NmsKind::Default
}

pub fn default_yolo_point() -> YoloPoint {
    YoloPoint::Center
}

// pub fn default_jitter() -> R64 {
//     R64::new(0.2)
// }

// pub fn default_ignore_thresh() -> R64 {
//     R64::new(0.5)
// }

// pub fn default_iou_normalizer() -> R64 {
//     R64::new(0.75)
// }
