use super::*;
use crate::common::*;

fn cal_iou_tlbr(bbox_a: (R64, R64, R64, R64), bbox_b: (R64, R64, R64, R64)) -> R64 {
    let xa = std::cmp::max(bbox_a.0, bbox_b.0);
    let ya = std::cmp::max(bbox_a.1, bbox_b.1);
    let xb = std::cmp::min(bbox_a.2, bbox_b.2);
    let yb = std::cmp::min(bbox_a.3, bbox_b.3);
    let inter_area =
        std::cmp::max(r64(0.0), xb - xa + r64(1.0)) * std::cmp::max(r64(0.0), yb - ya + r64(1.0));
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
    fn get_tlbr(&self) -> (R64, R64, R64, R64) {
        (*self).get_tlbr()
    }
    fn confidence(&self) -> R64 {
        (*self).confidence()
    }
    fn get_cls_conf(&self) -> R64 {
        (*self).get_cls_conf()
    }
    fn get_cls_id(&self) -> i32 {
        (*self).get_cls_id()
    }
    fn get_id(&self) -> i32 {
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

impl PredBox for MDetection {
    fn get_tlbr(&self) -> (R64, R64, R64, R64) {
        (self.x1, self.y1, self.x2, self.y2)
    }
    fn confidence(&self) -> R64 {
        self.conf
    }
    fn get_cls_conf(&self) -> R64 {
        self.cls_conf
    }
    fn get_cls_id(&self) -> i32 {
        self.cls_id
    }
    fn get_id(&self) -> i32 {
        self.id
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetBoxStruct {
    detection: MDetection,
    ground_truth: Option<MDetection>,
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

impl DetBoxStruct {
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
            detection: MDetection::new(d_id, d_x1, d_y1, d_x2, d_y2, d_conf, d_cls_conf, d_cls_id),
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
            detection: MDetection::new(d_id, d_x1, d_y1, d_x2, d_y2, d_conf, d_cls_conf, d_cls_id),
            ground_truth: None,
        }
    }
}

//impl DetectionForAp<D, G>

pub fn match_det_gt<I, T>(dets: I, gts: I) -> Vec<DetBoxStruct>
where
    I: IntoIterator<Item = T>,
    T: PredBox,
{
    let dets: Vec<_> = dets.into_iter().collect();
    let gts: Vec<_> = gts.into_iter().collect();

    let ret: Vec<_> = dets
        .iter()
        .map(|det| {
            let d_id = det.get_id();
            let d_box = det.get_tlbr();
            let d_conf = det.confidence();
            let d_cls_conf = det.get_cls_conf();
            let d_cls_id = det.get_cls_id();
            let gt_iou_boxes: Vec<_> = gts
                .iter()
                .map(|gt| {
                    let g_id = gt.get_id();
                    let g_cls_id = gt.get_cls_id();
                    let g_box = gt.get_tlbr();
                    let iou = cal_iou_tlbr(d_box, g_box);
                    (iou, g_id, g_cls_id, g_box)
                })
                .collect();
            let gt_iou_max = gt_iou_boxes.iter().max();

            let (iou, g_id, g_cls_id, g_box) = gt_iou_max.unwrap();

            let mut paired: DetBoxStruct;

            if *iou == r64(0.0) {
                //no matching gt
                let (d_x1, d_y1, d_x2, d_y2) = d_box;
                paired = DetBoxStruct::new_no_gt(
                    d_id, d_x1, d_y1, d_x2, d_y2, d_conf, d_cls_conf, d_cls_id,
                );
            } else {
                let (d_x1, d_y1, d_x2, d_y2) = d_box;
                let (g_x1, g_y1, g_x2, g_y2) = g_box;
                paired = DetBoxStruct::new(
                    d_id, d_x1, d_y1, d_x2, d_y2, d_conf, d_cls_conf, d_cls_id, *g_id, *g_cls_id,
                    *g_x1, *g_y1, *g_x2, *g_y2,
                );
            }
            paired
        })
        .collect();
    ret
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cal_iou_xxyys(bbox_a: (R64, R64, R64, R64), bbox_b: (R64, R64, R64, R64)) -> R64 {
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

    fn match_d_g(dets: &[MDetection], gts: &[MDetection]) -> Vec<DetBoxStruct> {
        let mut bbox_d: (R64, R64, R64, R64);
        let mut bbox_g: (R64, R64, R64, R64);
        let mut max_iou: R64;
        let mut tmp_iou: R64;
        let mut sel_g: usize;
        let mut td: DetBoxStruct;
        let mut t_vec: Vec<DetBoxStruct> = Vec::new();
        for d_id in 0..dets.len() {
            max_iou = r64(0.0);
            sel_g = 0;
            bbox_d = dets[d_id].get_tlbr();
            for g_id in 0..gts.len() {
                bbox_g = gts[g_id].get_tlbr();
                tmp_iou = cal_iou_xxyys(bbox_d, bbox_g);
                if gts[g_id].cls_id == dets[d_id].cls_id && max_iou < tmp_iou {
                    max_iou = tmp_iou;
                    sel_g = g_id;
                }
            }
            if max_iou == r64(0.0) {
                td = DetBoxStruct::new_no_gt(
                    d_id as i32,
                    dets[d_id].x1,
                    dets[d_id].y1,
                    dets[d_id].x2,
                    dets[d_id].y2,
                    dets[d_id].conf,
                    dets[d_id].cls_conf,
                    dets[d_id].cls_id,
                );
            } else {
                td = DetBoxStruct::new(
                    d_id as i32,
                    dets[d_id].x1,
                    dets[d_id].y1,
                    dets[d_id].x2,
                    dets[d_id].y2,
                    dets[d_id].conf,
                    dets[d_id].cls_conf,
                    dets[d_id].cls_id,
                    sel_g as i32,
                    gts[sel_g].cls_id,
                    gts[sel_g].x1,
                    gts[sel_g].y1,
                    gts[sel_g].x2,
                    gts[sel_g].y2,
                );
            }
            t_vec.push(td);
        }
        t_vec
    }

    fn vecd_to_mdetection(in_vec: Vec<Vec<f64>>) -> Vec<MDetection> {
        let mut v_det: Vec<MDetection> = Vec::new();
        let mut mdt: MDetection;
        for i in 0..in_vec.len() {
            mdt = MDetection::new_f64(
                i as i32,
                in_vec[i][0],
                in_vec[i][1],
                in_vec[i][2],
                in_vec[i][3],
                in_vec[i][4],
                in_vec[i][5],
                in_vec[i][6] as i32,
            );
            v_det.push(mdt);
        }

        v_det
    }
    fn vecg_to_mdetection(in_vec: Vec<Vec<f64>>) -> Vec<MDetection> {
        let mut v_det: Vec<MDetection> = Vec::new();
        let mut mdt: MDetection;
        for i in 0..in_vec.len() {
            mdt = MDetection::new_f64(
                i as i32,
                in_vec[i][1],
                in_vec[i][2],
                in_vec[i][3],
                in_vec[i][4],
                1.0,
                1.0,
                in_vec[i][0] as i32,
            );
            v_det.push(mdt);
        }

        v_det
    }

    fn split_detection_class(vec_det: &Vec<DetBoxStruct>) -> Vec<Vec<DetBoxStruct>> {
        let mut vec_ret: Vec<Vec<DetBoxStruct>> = Vec::new();
        let mut cls_id_to_vec_id: Vec<i32> = Vec::new();
        let mut cls_id: i32;
        let mut vec_tmp: Vec<DetBoxStruct>;
        for i in 0..vec_det.len() {
            cls_id = vec_det[i].detection.cls_id;
            if cls_id_to_vec_id.iter().any(|&k| k == cls_id) {
                //The class is seen

                let index = cls_id_to_vec_id.iter().position(|&r| r == cls_id).unwrap();
                vec_ret[index].push(vec_det[i]);
            } else {
                cls_id_to_vec_id.push(cls_id);
                vec_tmp = Vec::new();
                vec_tmp.push(vec_det[i]);
                vec_ret.push(vec_tmp);
            }
        }

        //dbg!(&vec_ret);
        vec_ret
    }
    //(cls_id, num of gt)
    fn get_gt_cnt_per_class(m_gt_vec: &Vec<MDetection>) -> Vec<(i32, usize)> {
        let mut vec_ret: Vec<(i32, usize)> = Vec::new();
        let mut class_seen: Vec<i32> = Vec::new();
        for i in 0..m_gt_vec.len() {
            let cls_id = m_gt_vec[i].cls_id;
            if class_seen.iter().any(|&k| k == cls_id) {
                //The class is seen
                let index = class_seen.iter().position(|&r| r == cls_id).unwrap();
                vec_ret[index].1 += 1;
            } else {
                class_seen.push(cls_id);
                vec_ret.push((cls_id, 1));
            }
        }
        //dbg!(&vec_ret);
        vec_ret
    }

    #[test]
    fn t_match_det_gt() -> Result<()> {
        let text = "0.00000 227.16200 219.68274 312.70200 410.39253
0.00000 284.18624 189.21947 335.15290 404.17874
0.00000 0.60445 237.66579 24.34890 415.77453
0.00000 174.27155 155.53200 246.64890 359.78800
34.00000 8.58000 330.53821 31.98000 411.12074";

        let text_d = "175.30000 170.77000 245.34000 324.72000 0.99968 0.99998 0.00000
284.07000 191.51000 336.73000 351.94000 0.98834 0.99999 0.00000
229.29000 222.98000 314.37000 358.82000 0.98327 0.99990 0.00000
0.35714 234.53000 29.80900 361.46000 0.89682 0.99831 0.00000";
        let d_vec: Result<Vec<Vec<f64>>> = text_d
            .lines()
            .map(|line| {
                let values: Result<Vec<_>> = line
                    .split(" ")
                    .map(|token| -> Result<_> {
                        let value: f64 = token.parse()?;
                        Ok(value)
                    })
                    .collect();
                values
            })
            .collect();
        let d_vec = d_vec?;
        let m_d_vec = vecd_to_mdetection(d_vec);

        let gt_vec: Result<Vec<Vec<f64>>> = text
            .lines()
            .map(|line| {
                let values: Result<Vec<_>> = line
                    .split(" ")
                    .map(|token| -> Result<_> {
                        let value: f64 = token.parse()?;
                        Ok(value)
                    })
                    .collect();
                values
            })
            .collect();
        let gt_vec = gt_vec?;
        let m_gt_vec = vecg_to_mdetection(gt_vec);
        let gt_cnt = get_gt_cnt_per_class(&m_gt_vec);
        let t_vec: Vec<DetBoxStruct>;
        let nt_vec: Vec<DetBoxStruct>;
        t_vec = match_d_g(&m_d_vec, &m_gt_vec);
        nt_vec = match_det_gt(&m_d_vec, &m_gt_vec);
        assert_eq!(t_vec, nt_vec);
        Ok(())
    }
}
