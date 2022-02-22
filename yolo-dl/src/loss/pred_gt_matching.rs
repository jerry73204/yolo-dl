use super::*;
use crate::common::*;
use bbox::{prelude::*, TLBR};

const EPSILON: f64 = 1e-8;

/// Trait for predicted bounding boxes
pub trait PredBox {
    fn tlbr(&self) -> TLBR<R64>;
    fn confidence(&self) -> R64;
    fn cls_conf(&self) -> R64;
    fn cls_id(&self) -> i32;
    fn id(&self) -> i32;

    fn to_mdetection(&self) -> MDetection {
        MDetection {
            id: self.id(),
            tlbr: self.tlbr(),
            conf: self.confidence(),
            cls_conf: self.cls_conf(),
            cls_id: self.cls_id(),
        }
    }
}

impl<T> PredBox for &T
where
    T: PredBox,
{
    fn tlbr(&self) -> TLBR<R64> {
        (*self).tlbr()
    }

    fn confidence(&self) -> R64 {
        (*self).confidence()
    }

    fn cls_conf(&self) -> R64 {
        (*self).cls_conf()
    }

    fn cls_id(&self) -> i32 {
        (*self).cls_id()
    }

    fn id(&self) -> i32 {
        (*self).id()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MDetection {
    pub id: i32,
    pub tlbr: TLBR<R64>,
    pub conf: R64,
    pub cls_conf: R64,
    pub cls_id: i32,
}

impl PredBox for MDetection {
    fn tlbr(&self) -> TLBR<R64> {
        self.tlbr.clone()
    }
    fn confidence(&self) -> R64 {
        self.conf
    }
    fn cls_conf(&self) -> R64 {
        self.cls_conf
    }
    fn cls_id(&self) -> i32 {
        self.cls_id
    }
    fn id(&self) -> i32 {
        self.id
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DetBoxStruct {
    pub detection: MDetection,
    pub ground_truth: Option<MDetection>,
}

impl average_precision::DetectionForAp<MDetection, MDetection> for DetBoxStruct {
    fn detection(&self) -> &MDetection {
        &self.detection
    }

    fn ground_truth(&self) -> Option<&MDetection> {
        self.ground_truth.as_ref()
    }

    fn confidence(&self) -> R64 {
        self.detection.cls_conf
    }

    fn iou(&self) -> R64 {
        self.ground_truth
            .as_ref()
            .map(|ground_truth| {
                ground_truth
                    .tlbr()
                    .iou_with(&self.detection.tlbr(), r64(EPSILON))
            })
            .unwrap_or_else(|| r64(0.0))
    }
}

pub fn match_det_gt<T>(
    dets: impl IntoIterator<Item = T>,
    gts: impl IntoIterator<Item = T>,
) -> Vec<DetBoxStruct>
where
    T: PredBox,
{
    let gts: Vec<_> = gts.into_iter().collect();

    let ret: Vec<_> = dets
        .into_iter()
        .map(|det| {
            // find ground truth with max IoU
            let (gt, iou) = gts
                .iter()
                .map(|gt| {
                    let iou = det.tlbr().iou_with(&gt.tlbr(), r64(EPSILON));
                    (gt, iou)
                })
                .max_by_key(|(_gt, iou)| *iou)
                .unwrap();

            let detection = det.to_mdetection();
            let ground_truth = (!abs_diff_eq!(iou.raw(), 0.0)).then(|| gt.to_mdetection());

            DetBoxStruct {
                detection,
                ground_truth,
            }
        })
        .collect();

    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_match_det_gt() -> Result<()> {
        // raw data
        let text_gts = "0 227.16200 219.68274 312.70200 410.39253
0 284.18624 189.21947 335.15290 404.17874
0 0.60445 237.66579 24.34890 415.77453
0 174.27155 155.53200 246.64890 359.78800
34 8.58000 330.53821 31.98000 411.12074";

        let text_dets = "175.30000 170.77000 245.34000 324.72000 0.99968 0.99998 0
284.07000 191.51000 336.73000 351.94000 0.98834 0.99999 0
229.29000 222.98000 314.37000 358.82000 0.98327 0.99990 0
0.35714 234.53000 29.80900 361.46000 0.89682 0.99831 0";

        // parse detections
        let dets: Vec<_> = text_dets
            .lines()
            .enumerate()
            .map(|(id, line)| -> Result<_> {
                let mut tokens = line.split(' ');
                let t: f64 = tokens.next().unwrap().parse()?;
                let l: f64 = tokens.next().unwrap().parse()?;
                let b: f64 = tokens.next().unwrap().parse()?;
                let r: f64 = tokens.next().unwrap().parse()?;
                let conf: f64 = tokens.next().unwrap().parse()?;
                let cls_conf: f64 = tokens.next().unwrap().parse()?;
                let cls_id: i32 = tokens.next().unwrap().parse()?;

                Ok(MDetection {
                    id: id as i32,
                    tlbr: TLBR::from_tlbr([t, l, b, r]).cast(),
                    conf: r64(conf),
                    cls_conf: r64(cls_conf),
                    cls_id,
                })
            })
            .try_collect()?;

        // parse ground truths
        let gts: Vec<_> = text_gts
            .lines()
            .enumerate()
            .map(|(id, line)| -> Result<_> {
                let mut tokens = line.split(' ');
                let cls_id: i32 = tokens.next().unwrap().parse()?;
                let t: f64 = tokens.next().unwrap().parse()?;
                let l: f64 = tokens.next().unwrap().parse()?;
                let b: f64 = tokens.next().unwrap().parse()?;
                let r: f64 = tokens.next().unwrap().parse()?;

                Ok(MDetection {
                    id: id as i32,
                    tlbr: TLBR::from_tlbr([t, l, b, r]).cast(),
                    conf: r64(1.0),
                    cls_conf: r64(1.0),
                    cls_id,
                })
            })
            .try_collect()?;

        // count # of boxes per class
        let _gt_cnt: Vec<(i32, usize)> = gts
            .iter()
            .map(|gt| gt.cls_id)
            .collect::<counter::Counter<_>>()
            .into_map()
            .into_iter()
            .collect();

        let t_vec: Vec<DetBoxStruct> = dets
            .iter()
            .map(|det| {
                let (gt, iou) = gts
                    .iter()
                    .filter(|gt| det.cls_id == gt.cls_id)
                    .map(|gt| {
                        let iou = det.tlbr().iou_with(&gt.tlbr(), r64(EPSILON));
                        (gt, iou)
                    })
                    .max_by_key(|(_gt, iou)| *iou)
                    .unwrap();

                DetBoxStruct {
                    detection: det.clone(),
                    ground_truth: (!abs_diff_eq!(iou.raw(), 0.0)).then(|| gt.clone()),
                }
            })
            .collect();
        let nt_vec: Vec<DetBoxStruct> = match_det_gt(&dets, &gts);

        assert_eq!(t_vec, nt_vec);
        Ok(())
    }
}
