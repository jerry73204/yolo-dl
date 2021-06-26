use crate::{
    common::*,
    utils::{self, AsXY},
};

pub trait DetectionForAp<D, G> {
    fn detection(&self) -> &D;
    fn ground_truth(&self) -> Option<&G>;
    fn confidence(&self) -> R64;
    fn iou(&self) -> R64;
}

impl<T, D, G> DetectionForAp<D, G> for &T
where
    T: DetectionForAp<D, G>,
{
    fn detection(&self) -> &D {
        (*self).detection()
    }

    fn ground_truth(&self) -> Option<&G> {
        (*self).ground_truth()
    }

    fn confidence(&self) -> R64 {
        (*self).confidence()
    }

    fn iou(&self) -> R64 {
        (*self).iou()
    }
}

#[derive(Debug, Clone)]
pub struct PrecRec<T>
where
    T: Copy,
{
    pub precision: T,
    pub recall: T,
}

impl<T> AsXY<T, T> for PrecRec<T>
where
    T: Copy,
{
    fn x(&self) -> T {
        self.recall
    }

    fn y(&self) -> T {
        self.precision
    }
}

#[derive(Debug, Clone, Copy)]
pub enum IntegralMethod {
    Continuous,
    Interpolation(usize),
}

#[derive(Debug)]
pub struct ApCalculator {
    integral_method: IntegralMethod,
}

impl ApCalculator {
    pub fn new_coco() -> Self {
        Self::new(IntegralMethod::Interpolation(101)).unwrap()
    }

    pub fn new(integral_method: IntegralMethod) -> Result<Self> {
        if let IntegralMethod::Interpolation(n_points) = integral_method {
            ensure!(
                n_points >= 1,
                "invalid number of interpolated points {}",
                n_points
            );
        }

        Ok(Self { integral_method })
    }

    /// Compute average precision from a precision/recall curve.
    ///
    /// The input precision/recall list must be ordered by non-decreasing recall.
    pub fn compute_by_prec_rec(&self, sorted_prec_rec: &[impl Borrow<PrecRec<R64>>]) -> R64 {
        // compute precision envelope
        let enveloped = {
            let max_recall = sorted_prec_rec.last().unwrap().borrow().recall;
            let first = PrecRec {
                precision: r64(0.0),
                recall: r64(0.0),
            };
            let last = PrecRec {
                precision: r64(0.0),
                recall: (max_recall + 1e-3).min(r64(1.0)),
            };

            // append/prepend sentinel values
            let iter = {
                iter::once(&first)
                    .chain(sorted_prec_rec.iter().map(Borrow::borrow))
                    .chain(iter::once(&last))
            };

            let mut list: Vec<PrecRec<R64>> = vec![];

            for &PrecRec { precision, recall } in iter.rev() {
                match list.last_mut() {
                    Some(last) => {
                        let max_precision = last.precision.max(precision);

                        if last.recall == recall {
                            last.precision = last.precision.max(precision);
                        } else {
                            list.push(PrecRec {
                                recall,
                                precision: max_precision,
                            });
                        }
                    }
                    None => list.push(PrecRec { precision, recall }),
                }
            }
            /*
                            let mut list: Vec<_> = iter
                                .rev()
                                .scan(r64(0.0), |max_prec: &mut R64, prec_rec: &PrecRec<R64>| {
                                    let PrecRec { precision, recall } = *prec_rec;
                                    *max_prec = (*max_prec).max(precision);
                                    Some(PrecRec { recall, precision: *max_prec })
                                })
                                .collect();
            */
            list.reverse();
            list
        };

        // compute ap
        match self.integral_method {
            IntegralMethod::Interpolation(n_points) => {
                let points_iter =
                    (0..n_points).map(|index| r64(index as f64 / (n_points - 1) as f64));

                let interpolated: Vec<_> =
                    utils::interpolate_stepwise_values(points_iter, &enveloped)
                        .into_iter()
                        .map(|(recall, precision)| PrecRec { recall, precision })
                        .collect();
                let ap = interpolated
                    .into_iter()
                    .map(|prec_rec| prec_rec.precision)
                    .sum::<R64>()
                    / r64(n_points as f64);
                ap
            }
            IntegralMethod::Continuous => {
                todo!();
            }
        }
    }

    pub fn compute_by_detections<I, T, D, G>(
        &self,
        dets: I, // I = Vec<test_xxx>
        num_ground_truth: usize,
        iou_thresh: R64,
    ) -> R64
    where
        I: IntoIterator<Item = T>,
        T: DetectionForAp<D, G>,
        G: Eq + Hash,
    {
        let dets: Vec<_> = dets.into_iter().collect();

        // group by ground truth and each group sorts by decreasing IoU
        let det_groups = dets
            .iter()
            .map(|det| (det.ground_truth(), det))
            .into_group_map();

        // Mark true positive (tp) for every first detection with IoU >= threshold
        // for every ground truth
        let mut dets: Vec<_> = det_groups
            .into_iter()
            .flat_map(|(_ground_truth, mut dets)| {
                dets.sort_by_cached_key(|det| -det.iou());
                let mut dets = dets.into_iter();

                let first = dets.next().into_iter().map(|det| {
                    let is_tp = det.iou() >= iou_thresh;
                    (det, is_tp)
                });
                let remaining = dets.map(|det| (det, false));
                let iter = first.chain(remaining);
                iter
            })
            .collect();

        // sort by decreasing confidence
        dets.sort_by_key(|(det, _is_tp)| -det.confidence());

        // compute precision and recall, it is ordered by increasing recall automatically
        let prec_rec: Vec<_> = dets
            .into_iter()
            .scan((0, 0), |(acc_tp, acc_fp), (_det, is_tp)| {
                if is_tp {
                    *acc_tp += 1;
                } else {
                    *acc_fp += 1;
                }
                let prec_rec = {
                    let acc_tp = r64(*acc_tp as f64);
                    let acc_fp = r64(*acc_fp as f64);
                    PrecRec {
                        precision: acc_tp / (acc_tp + acc_fp),
                        recall: acc_tp / num_ground_truth as f64,
                    }
                };
                Some(prec_rec)
            })
            .collect();

        // compute ap
        let ap = self.compute_by_prec_rec(&prec_rec);
        ap
    }
}

#[derive(Debug)]
pub struct MeanApCalculator {
    ap_calculator: ApCalculator,
    iou_thresholds: Vec<R64>,
}

impl MeanApCalculator {
    pub fn new_coco() -> Self {
        let iou_thresholds: Vec<_> = (0..10)
            .map(|ind| 0.5 + ind as f64 * 0.05)
            .map(r64)
            .collect();
        Self {
            ap_calculator: ApCalculator::new_coco(),
            iou_thresholds,
        }
    }

    pub fn new(integral_method: IntegralMethod, iou_thresholds: Vec<R64>) -> Result<Self> {
        ensure!(
            !iou_thresholds.is_empty(),
            "iou_thresholds must be non-empty"
        );

        Ok(Self {
            ap_calculator: ApCalculator::new(integral_method).unwrap(),
            iou_thresholds,
        })
    }

    pub fn compute_mean_ap<D, G>(
        &self,
        dets: &[impl DetectionForAp<D, G>],
        num_ground_truth: usize,
    ) -> R64
    where
        G: Eq + Hash,
    {
        let sum_ap: R64 = self
            .iou_thresholds
            .iter()
            .cloned()
            .map(|iou_thresh| {
                let ap = self.ap_calculator.compute_by_detections(
                    dets.iter(),
                    num_ground_truth,
                    iou_thresh,
                );
                ap
            })
            .sum();
        let mean_ap = sum_ap / self.iou_thresholds.len() as f64;
        mean_ap
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// structure of detection output
    /// (x1, y1): coordinates of top-left corner of object with scale 416x416
    /// (x2, y2): coordinates of bottom-right corner of object with scale 416x416

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
        fn get_xyxy(&self) -> (R64, R64, R64, R64) {
            (self.x1, self.y1, self.x2, self.y2)
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct TestDetection {
        detection: MDetection,
        ground_truth: Option<MDetection>,
    }
    impl DetectionForAp<MDetection, MDetection> for TestDetection {
        // add code here
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

    impl TestDetection {
        fn new_f64(
            d_id: i32,
            d_x1: f64,
            d_y1: f64,
            d_x2: f64,
            d_y2: f64,
            d_conf: f64,
            d_cls_conf: f64,
            d_cls_id: i32,
            g_id: i32,
            g_cls_id: i32,
            g_x1: f64,
            g_y1: f64,
            g_x2: f64,
            g_y2: f64,
        ) -> TestDetection {
            TestDetection {
                detection: MDetection::new_f64(
                    d_id, d_x1, d_y1, d_x2, d_y2, d_conf, d_cls_conf, d_cls_id,
                ),
                ground_truth: Some(MDetection::new_f64(
                    g_id, g_x1, g_y1, g_x2, g_y2, 1.0000, 1.0000, g_cls_id,
                )),
            }
        }
        fn new_no_gt_f64(
            d_id: i32,
            d_x1: f64,
            d_y1: f64,
            d_x2: f64,
            d_y2: f64,
            d_conf: f64,
            d_cls_conf: f64,
            d_cls_id: i32,
        ) -> TestDetection {
            TestDetection {
                detection: MDetection::new_f64(
                    d_id, d_x1, d_y1, d_x2, d_y2, d_conf, d_cls_conf, d_cls_id,
                ),
                ground_truth: None,
            }
        }
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
        ) -> TestDetection {
            TestDetection {
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
        ) -> TestDetection {
            TestDetection {
                detection: MDetection::new(
                    d_id, d_x1, d_y1, d_x2, d_y2, d_conf, d_cls_conf, d_cls_id,
                ),
                ground_truth: None,
            }
        }
    }
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

    fn match_d_g(dets: &[MDetection], gts: &[MDetection]) -> Vec<TestDetection> {
        let mut bbox_d: (R64, R64, R64, R64);
        let mut bbox_g: (R64, R64, R64, R64);
        let mut max_iou: R64;
        let mut tmp_iou: R64;
        let mut sel_g: usize;
        let mut td: TestDetection;
        let mut t_vec: Vec<TestDetection> = Vec::new();
        for d_id in 0..dets.len() {
            max_iou = r64(0.0);
            sel_g = 0;
            bbox_d = dets[d_id].get_xyxy();
            for g_id in 0..gts.len() {
                bbox_g = gts[g_id].get_xyxy();
                tmp_iou = cal_iou_xxyys(bbox_d, bbox_g);
                if gts[g_id].cls_id == dets[d_id].cls_id && max_iou < tmp_iou {
                    max_iou = tmp_iou;
                    sel_g = g_id;
                }
            }
            if max_iou == r64(0.0) {
                td = TestDetection::new_no_gt(
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
                td = TestDetection::new(
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

    fn split_detection_class(vec_det: &Vec<TestDetection>) -> Vec<Vec<TestDetection>> {
        let mut vec_ret: Vec<Vec<TestDetection>> = Vec::new();
        let mut cls_id_to_vec_id: Vec<i32> = Vec::new();
        let mut cls_id: i32;
        let mut vec_tmp: Vec<TestDetection>;
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
    fn t_compute_by_detections() -> Result<()> {
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
        let t_vec: Vec<TestDetection>;
        t_vec = match_d_g(&m_d_vec, &m_gt_vec);
        let class_split_vec = split_detection_class(&t_vec);

        let ap_cal = ApCalculator::new_coco();
        let ret = ap_cal.compute_by_detections(t_vec, 4, R64::new(0.5));
        assert_eq!(ret, r64(1.0));
        Ok(())
    }

    #[test]
    fn t_mean_average_precision_cal() -> Result<()> {
        let text = "39.00000 61.40888 27.67710 141.49845 230.31445
56.00000 0.22360 92.69645 58.11374 148.82400
56.00000 144.48242 43.56290 416.00021 231.43224
60.00000 0.00000 137.03310 412.75354 410.12421
40.00000 160.14066 101.55579 245.92610 240.79890";

        let text_d = "159.15750 105.84630 247.27790 245.03130 0.99870 0.99960 40.00000
55.24000 31.11770 150.80330 362.72990 0.99670 0.99930 39.00000
200.69280 35.67050 411.24700 206.84590 0.78630 0.97070 56.00000";
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
        let t_vec: Vec<TestDetection>;
        t_vec = match_d_g(&m_d_vec, &m_gt_vec);
        let class_split_vec = split_detection_class(&t_vec);
        let mut sum_ret = r64(0.0);
        let map_cal = MeanApCalculator::new_coco();
        for i in 0..class_split_vec.len() {
            let cls_id = class_split_vec[i][0].detection.cls_id;
            let mut num_gt: usize = 0;
            for gt_cnt_cls in gt_cnt.iter() {
                if gt_cnt_cls.0 == cls_id {
                    num_gt = gt_cnt_cls.1;
                    break;
                }
            }

            let ret = map_cal.compute_mean_ap(&class_split_vec[i], num_gt);
            sum_ret += ret;
        }
        let map = sum_ret / gt_cnt.len() as f64;
        assert!(abs_diff_eq!(
            map.raw(),
            (0.9 + 0.1 + 0.19801980198019803) / gt_cnt.len() as f64
        ));

        Ok(())
    }

    #[test]
    fn t_compute_by_prec_rec() -> Result<()> {
        let ap_cal_11 = ApCalculator::new(IntegralMethod::Interpolation(11))?;
        let ap_cal = ApCalculator::new_coco();
        let mut vec = vec![PrecRec {
            precision: r64(1.0),
            recall: r64(1.0),
        }];
        vec = vec.into_iter().rev().collect();
        let res = ap_cal.compute_by_prec_rec(&vec);
        assert_eq!(res, r64(1.0));

        vec = vec![
            PrecRec {
                precision: r64(0.5),
                recall: r64(0.625),
            },
            PrecRec {
                precision: r64(0.556),
                recall: r64(0.625),
            },
            PrecRec {
                precision: r64(0.625),
                recall: r64(0.625),
            },
            PrecRec {
                precision: r64(0.714),
                recall: r64(0.625),
            },
            PrecRec {
                precision: r64(0.833),
                recall: r64(0.625),
            },
            PrecRec {
                precision: r64(0.800),
                recall: r64(0.500),
            },
            PrecRec {
                precision: r64(0.750),
                recall: r64(0.375),
            },
            PrecRec {
                precision: r64(1.0),
                recall: r64(0.375),
            },
            PrecRec {
                precision: r64(1.0),
                recall: r64(0.250),
            },
            PrecRec {
                precision: r64(1.0),
                recall: r64(0.125),
            },
        ];

        vec = vec.into_iter().rev().collect();
        let res = ap_cal_11.compute_by_prec_rec(&vec);
        assert!(abs_diff_eq!(res.raw(), 0.5908181818181819));
        Ok(())
    }
}
