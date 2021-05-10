use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
#[derivative(Hash)]
#[serde(try_from = "RawYolo")]
pub struct Yolo {
    pub classes: usize,
    pub max_boxes: usize,
    pub max_delta: Option<R64>,
    pub counters_per_class: Option<Vec<usize>>,
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
    pub track_history_size: usize,
    pub sim_thresh: R64,
    pub dets_for_track: usize,
    pub dets_for_show: usize,
    pub track_ciou_norm: R64,
    pub embedding_layer: Option<LayerIndex>,
    pub map: Option<PathBuf>,
    pub anchors: Vec<(usize, usize)>,
    pub yolo_point: YoloPoint,
    pub iou_loss: IouLoss,
    pub nms_kind: NmsKind,
    pub common: Common,
}

impl Yolo {
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

impl TryFrom<RawYolo> for Yolo {
    type Error = Error;

    fn try_from(from: RawYolo) -> Result<Self, Self::Error> {
        let RawYolo {
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
        } = from;

        let mask = mask.unwrap_or_else(IndexSet::new);
        let anchors = match (num, anchors) {
            (0, None) => vec![],
            (num, None) => {
                warn!("num={} is inconsistent with actual number of anchors (0), the field is ignored", num);
                vec![]
            }
            (num, Some(anchors)) => {
                if anchors.len() != num as usize {
                    warn!("num={} is inconsistent with actual number of anchors ({}), the field is ignored", num, anchors.len());
                }

                let anchors: Option<Vec<_>> = mask
                    .into_iter()
                    .map(|index| anchors.get(index as usize).copied())
                    .collect();
                anchors.ok_or_else(|| format_err!("mask index exceeds total number of anchors"))?
            }
        };

        Ok(Yolo {
            classes,
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
            iou_thresh_kind,
            beta_nms,
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
            yolo_point,
            iou_loss,
            nms_kind,
            common,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
#[derivative(Hash)]
pub(super) struct RawYolo {
    #[serde(default = "defaults::classes")]
    pub classes: usize,
    #[serde(default = "defaults::num")]
    pub num: usize,
    #[derivative(Hash(hash_with = "hash_option_vec_indexset::<usize, _>"))]
    #[serde(with = "serde_::mask", default)]
    pub mask: Option<IndexSet<usize>>,
    #[serde(rename = "max", default = "defaults::max_boxes")]
    pub max_boxes: usize,
    pub max_delta: Option<R64>,
    #[serde(with = "serde_::opt_vec_usize", default)]
    pub counters_per_class: Option<Vec<usize>>,
    #[serde(default = "defaults::yolo_label_smooth_eps")]
    pub label_smooth_eps: R64,
    #[serde(default = "defaults::scale_x_y")]
    pub scale_x_y: R64,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
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
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
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
    pub embedding_layer: Option<LayerIndex>,
    pub map: Option<PathBuf>,
    #[serde(with = "serde_::anchors", default)]
    pub anchors: Option<Vec<(usize, usize)>>,
    #[serde(flatten)]
    pub common: Common,
}
