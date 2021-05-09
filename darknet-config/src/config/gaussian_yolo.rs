use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Derivative, Serialize, Deserialize)]
#[serde(try_from = "RawGaussianYolo")]
pub struct GaussianYolo {
    pub classes: u64,
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
    pub common: Common,
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

        let mask = mask.unwrap_or_else(IndexSet::new);
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
                anchors.ok_or_else(|| format_err!("mask index exceeds total number of anchors"))?
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
    #[serde(default = "defaults::classes")]
    pub classes: u64,
    #[serde(rename = "max", default = "defaults::max_boxes")]
    pub max_boxes: u64,
    #[serde(default = "defaults::num")]
    pub num: u64,
    #[derivative(Hash(hash_with = "hash_option_vec_indexset::<u64, _>"))]
    #[serde(with = "serde_::mask", default)]
    pub mask: Option<IndexSet<u64>>,
    pub max_delta: Option<R64>,
    #[serde(with = "serde_::opt_vec_u64", default)]
    pub counters_per_class: Option<Vec<u64>>,
    #[serde(default = "defaults::yolo_label_smooth_eps")]
    pub label_smooth_eps: R64,
    #[serde(default = "defaults::scale_x_y")]
    pub scale_x_y: R64,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
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
    #[serde(with = "serde_::anchors", default)]
    pub anchors: Option<Vec<(u64, u64)>>,
    #[serde(flatten)]
    pub common: Common,
}
