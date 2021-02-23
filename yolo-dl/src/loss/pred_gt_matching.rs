use crate::common::*;

fn cal_iou_tlbr(bbox_a: (R64, R64, R64, R64), bbox_b: (R64, R64, R64, R64)) -> R64 {
    let xa = std::cmp::max(bbox_a.0, bbox_b.0);
    let ya = std::cmp::max(bbox_a.1, bbox_b.1);
    let xb = std::cmp::min(bbox_a.2, bbox_b.2);
    let yb = std::cmp::min(bbox_a.3, bbox_b.3);
    let inter_area = std::cmp::max(r64(0.0), xb - xa + r64(1.0))
        * std::cmp::max(r64(0.0), yb - ya + r64(1.0));
    let box_a_area = (bbox_a.2 - bbox_a.0 + r64(1.0)) * (bbox_a.3 - bbox_a.1 + r64(1.0));
    let box_b_area = (bbox_b.2 - bbox_b.0 + r64(1.0)) * (bbox_b.3 - bbox_b.1 + r64(1.0));
    let iou = inter_area / (box_a_area + box_b_area - inter_area);
    iou
}

pub trait PredBox {
    fn get_tlbr(&self) -> (R64, R64, R64, R64);
    fn confidence(&self) -> R64;
    fn get_cls_conf(&self) -> R64;
    fn get_cls_id(&self) -> i32;
    fn get_id(&self) -> i32;
}

impl<T> PredBox for &T
where
    T: PredBox,
{
    fn get_tlbr(&self) -> (R64, R64, R64, R64){
        (*self).get_tlbr()
    }
    fn confidence(&self) -> R64{
        (*self).confidence()
    }
    fn get_cls_conf(&self) -> R64{
        (*self).get_cls_conf()
    }
    fn get_cls_id(&self) -> i32{
        (*self).get_cls_id()
    }
    fn get_id(&self) -> i32{
        (*self).get_id()
    }
}

#[derive(Debug, Clone, Eq, Copy)]
struct MDetection {
    id: i32,
    x1: R64,
    y1: R64,
    x2: R64,
    y2: R64,
    conf: R64,
    cls_conf: R64,
    cls_id: i32,
}
impl PartialEq for MDetection {
    fn eq(&self, other: &MDetection) -> bool {
        //self.id == other.id
        self.x1 == other.x1
            && self.x2 == other.x2
            && self.y1 == other.y1
            && self.y2 == other.y2
            && self.conf == other.conf
            && self.cls_conf == other.cls_conf
    }
}

impl Hash for MDetection {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        //self.phone.hash(state);
    }
}

impl MDetection {
    fn new_f64(
        id: i32,
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        conf: f64,
        cls_conf: f64,
        cls_id: i32,
    ) -> MDetection {
        MDetection {
            id,
            x1: r64(x1),
            y1: r64(y1),
            x2: r64(x2),
            y2: r64(y2),
            conf: r64(conf),
            cls_conf: r64(cls_conf),
            cls_id,
        }
    }
    fn new(
        id: i32,
        x1: R64,
        y1: R64,
        x2: R64,
        y2: R64,
        conf: R64,
        cls_conf: R64,
        cls_id: i32,
    ) -> MDetection {
        MDetection {
            id,
            x1,
            y1,
            x2,
            y2,
            conf,
            cls_conf,
            cls_id,
        }
    }
    fn get_tlbr(&self) -> (R64, R64, R64, R64) {
        (self.x1, self.y1, self.x2, self.y2)
    }
}


#[derive(Debug, Clone, Copy)]
pub struct DetBoxStruct{
    detection: MDetection,
    ground_truth: Option<MDetection>,
}
/*
impl average_precision::DetectionForAp<MDetection, MDetection> for DetBoxStruct {
    fn detection(&self) -> &MDetection {
        &self.detection
    }
    fn ground_truth(&self) -> Option<&MDetection> {
        self.ground_truth.as_ref()
    }
    fn confidence(&self) -> R64 {
        self.detection.cls_conf
        //self.detection.conf*self.detection.cls_conf
    }
    fn iou(&self) -> R64 {
        match &self.ground_truth {
            None => r64(0.0),
            Some(gt) => {
                let xa = std::cmp::max(self.detection.x1, gt.x1);
                let ya = std::cmp::max(self.detection.y1, gt.y1);
                let xb = std::cmp::min(self.detection.x2, gt.x2);
                let yb = std::cmp::min(self.detection.y2, gt.y2);
                let inter_area = std::cmp::max(r64(0.0), xb - xa + r64(1.0))
                    * std::cmp::max(r64(0.0), yb - ya + r64(1.0));
                let box_a_area = (self.detection.x2 - self.detection.x1 + r64(1.0))
                    * (self.detection.y2 - self.detection.y1 + r64(1.0));
                let box_b_area = (gt.x2 - gt.x1 + r64(1.0)) * (gt.y2 - gt.y1 + r64(1.0));
                let iou = inter_area / (box_a_area + box_b_area - inter_area);
                iou
            }
        }
    }
}
*/
impl DetBoxStruct{
    fn new(
        d_id: i32,
        d_x1: R64,
        d_y1: R64,
        d_x2: R64,
        d_y2: R64,
        d_conf: R64,
        d_cls_conf: R64,
        d_cls_id: i32,
        g_id: i32,
        g_cls_id: i32,
        g_x1: R64,
        g_y1: R64,
        g_x2: R64,
        g_y2: R64,
    ) -> DetBoxStruct {
        DetBoxStruct {
            detection: MDetection::new(
                d_id, d_x1, d_y1, d_x2, d_y2, d_conf, d_cls_conf, d_cls_id,
            ),
            ground_truth: Some(MDetection::new(
                g_id,
                g_x1,
                g_y1,
                g_x2,
                g_y2,
                r64(1.0000),
                r64(1.0000),
                g_cls_id,
            )),
        }
    }
    fn new_no_gt(
        d_id: i32,
        d_x1: R64,
        d_y1: R64,
        d_x2: R64,
        d_y2: R64,
        d_conf: R64,
        d_cls_conf: R64,
        d_cls_id: i32,
    ) -> DetBoxStruct {
        DetBoxStruct {
            detection: MDetection::new(
                d_id, d_x1, d_y1, d_x2, d_y2, d_conf, d_cls_conf, d_cls_id,
            ),
            ground_truth: None,
        }
    }
}

//impl DetectionForAp<D, G>

pub fn match_det_gt<I, T>(
    dets: I,
    gts: I,
) -> Vec<DetBoxStruct>
where
    I: IntoIterator<Item = T>,
    T: PredBox,
{
    let dets: Vec<_> = dets.into_iter().collect();
    let gts: Vec<_> = gts.into_iter().collect();

    let ret : Vec<_> = dets
        .iter()
        .map(|det| {
            let d_id = det.get_id();
            let d_box = det.get_tlbr();
            let d_conf = det.confidence();
            let d_cls_conf = det.get_cls_conf();
            let d_cls_id = det.get_cls_id();
            let gt_iou_boxes : Vec<_> = gts
                .iter()
                .map(|gt|{
                    let g_id = gt.get_id();
                    let g_cls_id = gt.get_cls_id();
                    let g_box = gt.get_tlbr();
                    let iou = cal_iou_tlbr(d_box, g_box);
                    (iou, g_id, g_cls_id, g_box)
                })
                .collect();
            let gt_iou_max = gt_iou_boxes
                .iter()
                .max();

            let (iou, g_id, g_cls_id, g_box) = gt_iou_max.unwrap();

            let mut paired : DetBoxStruct;

            if *iou == r64(0.0){
                //no matching gt
                let (d_x1, d_y1, d_x2, d_y2) = d_box;
                paired = DetBoxStruct::new_no_gt(d_id, d_x1, d_y1, d_x2, d_y2, d_conf, d_cls_conf, d_cls_id);
            }
            else{
                let (d_x1, d_y1, d_x2, d_y2) = d_box;
                let (g_x1, g_y1, g_x2, g_y2) = g_box;
                paired = DetBoxStruct::new(d_id, d_x1, d_y1, d_x2, d_y2, d_conf, d_cls_conf, d_cls_id, *g_id, *g_cls_id, *g_x1, *g_y1, *g_x2, *g_y2);
            }
            paired
        })
        .collect();
    ret
}
