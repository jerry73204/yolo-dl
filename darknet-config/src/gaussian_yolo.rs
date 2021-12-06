use super::{Anchors, IouLoss, IouThreshold, Meta, NmsKind, OutputShape, YoloPoint};
use crate::{common::*, utils};

#[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
#[derivative(Hash)]
pub struct GaussianYolo {
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
    #[serde(flatten)]
    pub common: Meta,
}

impl GaussianYolo {
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

// pub fn classes() -> usize {
//     warn!("classes option is not specified, use default 20");
//     20
// }

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
