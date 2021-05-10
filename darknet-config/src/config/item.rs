use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) enum Item {
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
    #[serde(rename = "batchnorm")]
    BatchNorm(BatchNorm),
    #[serde(rename = "dropout")]
    Dropout(Dropout),
    #[serde(rename = "softmax")]
    Softmax(Softmax),
    #[serde(rename = "Gaussian_yolo")]
    GaussianYolo(RawGaussianYolo),
    #[serde(rename = "yolo")]
    Yolo(RawYolo),
    #[serde(rename = "cost")]
    Cost(Cost),
    #[serde(rename = "crop")]
    Crop(Crop),
    #[serde(rename = "avgpool")]
    AvgPool(AvgPool),
    #[serde(rename = "local_avgpool")]
    LocalAvgPool(UnimplementedLayer),
    #[serde(rename = "crnn")]
    Crnn(UnimplementedLayer),
    #[serde(rename = "sam")]
    Sam(UnimplementedLayer),
    #[serde(rename = "scale_channels")]
    ScaleChannels(UnimplementedLayer),
    #[serde(rename = "gru")]
    Gru(UnimplementedLayer),
    #[serde(rename = "lstm")]
    Lstm(UnimplementedLayer),
    #[serde(rename = "rnn")]
    Rnn(UnimplementedLayer),
    #[serde(rename = "detection")]
    Detection(UnimplementedLayer),
    #[serde(rename = "region")]
    Region(UnimplementedLayer),
    #[serde(rename = "reorg")]
    Reorg(UnimplementedLayer),
    #[serde(rename = "contrastive")]
    Contrastive(UnimplementedLayer),
}

impl Item {
    pub fn try_into_net(self) -> Result<Net> {
        let net = match self {
            Self::Net(net) => net,
            _ => bail!("not a net layer"),
        };
        Ok(net)
    }

    pub fn try_into_layer_config(self) -> Result<Layer> {
        let layer = match self {
            Self::Connected(layer) => Layer::Connected(layer),
            Self::Convolutional(layer) => Layer::Convolutional(layer),
            Self::Route(layer) => Layer::Route(layer),
            Self::Shortcut(layer) => Layer::Shortcut(layer),
            Self::MaxPool(layer) => Layer::MaxPool(layer),
            Self::UpSample(layer) => Layer::UpSample(layer),
            Self::BatchNorm(layer) => Layer::BatchNorm(layer),
            Self::Dropout(layer) => Layer::Dropout(layer),
            Self::Softmax(layer) => Layer::Softmax(layer),
            Self::Cost(layer) => Layer::Cost(layer),
            Self::Crop(layer) => Layer::Crop(layer),
            Self::AvgPool(layer) => Layer::AvgPool(layer),
            Self::GaussianYolo(layer) => Layer::GaussianYolo(layer.try_into()?),
            Self::Yolo(layer) => Layer::Yolo(layer.try_into()?),
            Self::Net(_layer) => {
                bail!("the 'net' layer must appear in the first section")
            }
            // unimplemented
            Self::Crnn(layer)
            | Self::Sam(layer)
            | Self::ScaleChannels(layer)
            | Self::LocalAvgPool(layer)
            | Self::Contrastive(layer)
            | Self::Detection(layer)
            | Self::Region(layer)
            | Self::Reorg(layer)
            | Self::Rnn(layer)
            | Self::Lstm(layer)
            | Self::Gru(layer) => Layer::Unimplemented(layer),
        };
        Ok(layer)
    }
}

impl TryFrom<Darknet> for Items {
    type Error = Error;

    fn try_from(config: Darknet) -> Result<Self, Self::Error> {
        let Darknet {
            net,
            layers: orig_layers,
        } = config;

        // extract global options that will be placed into yolo layers
        let global_anchors: Vec<_> = orig_layers
            .iter()
            .filter_map(|layer| match layer {
                Layer::Yolo(yolo) => {
                    let Yolo { anchors, .. } = yolo;
                    Some(anchors)
                }
                Layer::GaussianYolo(yolo) => {
                    let GaussianYolo { anchors, .. } = yolo;
                    Some(anchors)
                }
                _ => None,
            })
            .flat_map(|anchors| anchors.iter().cloned())
            .collect();

        let items: Vec<_> = {
            let mut mask_count = 0;

            iter::once(Ok(Item::Net(net)))
                .chain(orig_layers.into_iter().map(|layer| -> Result<_> {
                    let item = match layer {
                        Layer::Connected(layer) => Item::Connected(layer),
                        Layer::Convolutional(layer) => Item::Convolutional(layer),
                        Layer::Route(layer) => Item::Route(layer),
                        Layer::Shortcut(layer) => Item::Shortcut(layer),
                        Layer::MaxPool(layer) => Item::MaxPool(layer),
                        Layer::UpSample(layer) => Item::UpSample(layer),
                        Layer::BatchNorm(layer) => Item::BatchNorm(layer),
                        Layer::Dropout(layer) => Item::Dropout(layer),
                        Layer::Softmax(layer) => Item::Softmax(layer),
                        Layer::Cost(layer) => Item::Cost(layer),
                        Layer::Crop(layer) => Item::Crop(layer),
                        Layer::AvgPool(layer) => Item::AvgPool(layer),
                        Layer::Unimplemented(_layer) => bail!("unimplemented layer"),
                        Layer::Yolo(orig_layer) => {
                            let Yolo {
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
                                anchors: local_anchors,
                                common,
                            } = orig_layer;

                            // build mask list
                            let mask: IndexSet<_> = {
                                let num_anchors = local_anchors.len();
                                let mask_begin = mask_count;
                                let mask_end = mask_begin + num_anchors;

                                // update counter
                                mask_count += num_anchors;

                                (mask_begin..mask_end).map(|index| index as usize).collect()
                            };

                            // make sure mask indexes are valid
                            assert!(
                                mask.iter()
                                    .cloned()
                                    .all(|index| (index as usize) < global_anchors.len()),
                                "mask indexes must not exceed total number of anchors"
                            );

                            let num = global_anchors.len() as usize;
                            let mask = if mask.is_empty() { None } else { Some(mask) };
                            let anchors = if global_anchors.is_empty() {
                                None
                            } else {
                                Some(global_anchors.clone())
                            };

                            Item::Yolo(RawYolo {
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
                            })
                        }
                        Layer::GaussianYolo(orig_layer) => {
                            let GaussianYolo {
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
                                anchors: local_anchors,
                                common,
                            } = orig_layer;

                            // build mask list
                            let mask: IndexSet<_> = {
                                let num_anchors = local_anchors.len();
                                let mask_begin = mask_count;
                                let mask_end = mask_begin + num_anchors;

                                // update counter
                                mask_count += num_anchors;

                                (mask_begin..mask_end).map(|index| index as usize).collect()
                            };

                            // make sure mask indexes are valid
                            assert!(
                                mask.iter()
                                    .cloned()
                                    .all(|index| (index as usize) < global_anchors.len()),
                                "mask indexes must not exceed total number of anchors"
                            );

                            let num = global_anchors.len() as usize;
                            let mask = if mask.is_empty() { None } else { Some(mask) };
                            let anchors = if global_anchors.is_empty() {
                                None
                            } else {
                                Some(global_anchors.clone())
                            };

                            Item::GaussianYolo(RawGaussianYolo {
                                classes,
                                max_boxes,
                                num,
                                mask,
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
                            })
                        }
                    };

                    Ok(item)
                }))
                .try_collect()?
        };

        Ok(Items(items))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub(super) struct Items(pub Vec<Item>);
