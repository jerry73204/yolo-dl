use anyhow::{format_err, Result};
use noisy_float::prelude::*;

/// Jerry's comments here
///
/// - Structs, enums, functions can be annotated with "doc comments", it starts with `///`.
/// - No parentheses in the condition of `if condition { .. }`.
/// - Try to use iterators if possible
/// - To break more than one layer of loops, mark the loop with name`'name loop {}` and break by name `break 'name`.
/// - The last statement of function can be `value` instead of `return value;`.
/// - We suggest `#[derive(Debug, Clone)]` to make type clonable and can be printed.
/// - `#[derive(PartialOrd, Ord)]` makes a type comparable, not that `f64` does not have `Ord`, use `R64` instead.
///
/// This is Jerry's impl.
mod jerry {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct PrecRec {
        pub prec: f64,
        pub rec: f64,
    }

    impl PrecRec {
        pub fn to_checked(&self) -> Option<PrecRecChecked> {
            let Self { prec, rec } = *self;
            let prec = R64::try_new(prec)?;
            let rec = R64::try_new(rec)?;

            if !(prec >= 0.0 && prec <= 1.0 && rec >= 0.0 && rec <= 1.0) {
                return None;
            }

            Some(PrecRecChecked { prec, rec })
        }
    }

    #[derive(Debug, Clone)]
    pub struct PrecRecChecked {
        pub prec: R64,
        pub rec: R64,
    }

    fn calc_map(input: &[PrecRec], num_points: usize) -> Result<()> {
        // sanity check
        let checked: Option<Vec<_>> = input.iter().map(|pair| pair.to_checked()).collect();
        let mut checked = checked.ok_or_else(|| format_err!("invalid input"))?;

        // sort by recall then precision
        checked.sort_by_cached_key(|pair| (pair.rec, pair.prec));

        // compute stepwise precision
        let stepwise = to_stepwise(&checked);

        // interpolation
        todo!("interpolate values by 'num_points' positions");
    }

    /// Convert to "stepwise" precision.
    ///
    /// The input is assumed to be sorted by recall.
    fn to_stepwise(input: &[PrecRecChecked]) -> Vec<PrecRecChecked> {
        let max_prec_reversed: Vec<R64> = input
            .iter()
            .rev()
            .map(|prec_rec| prec_rec.prec)
            .scan(None, |prev_max: &mut Option<R64>, curr| {
                let curr_max = prev_max.map(|prev| prev.max(curr)).unwrap_or(curr);
                *prev_max = Some(curr_max);
                Some(curr_max)
            })
            .collect();

        let output: Vec<_> = input
            .iter()
            .zip(max_prec_reversed.into_iter().rev())
            .map(|(prec_rec, new_prec)| PrecRecChecked {
                prec: new_prec,
                rec: prec_rec.rec,
            })
            .collect();

        output
    }
}

/// Vincent's impl
mod vincent {
    use super::*;

    #[derive(Debug, Clone)]
    struct PrTable {
        pr_arr: [f64; 101], //This stores the 101-interpolated Precision-Recall
        iou: i32,
    }

    impl PrTable {
        fn new(pr_init_val: f64, iou: i32) -> PrTable {
            PrTable {
                pr_arr: [pr_init_val; 101],
                iou: iou,
            }
        }
    }

    /// AvgPrec Stores the AP calculated along with the IOU, the actual iou stored is the original iou * 100
    #[derive(Debug, Clone)]
    struct AvgPrec {
        avg_prec: f64,
        iou: i32,
    }

    impl AvgPrec {
        fn new(ap_val: f64, iou: i32) -> AvgPrec {
            AvgPrec {
                avg_prec: ap_val,
                iou,
            }
        }
    }

    /// ApTables stores the AP of IOU from 0.5 to 0.95
    /// Each class has one ApTables
    #[derive(Debug, Clone)]
    struct ApTables {
        table_arr: [AvgPrec; 10],
    }

    impl ApTables {
        fn new(
            i1: f64,
            i2: f64,
            i3: f64,
            i4: f64,
            i5: f64,
            i6: f64,
            i7: f64,
            i_8: f64,
            i9: f64,
            i10: f64,
        ) -> ApTables {
            ApTables {
                table_arr: [
                    AvgPrec::new(i1, 50),
                    AvgPrec::new(i2, 55),
                    AvgPrec::new(i3, 60),
                    AvgPrec::new(i4, 65),
                    AvgPrec::new(i5, 70),
                    AvgPrec::new(i6, 75),
                    AvgPrec::new(i7, 80),
                    AvgPrec::new(i_8, 85),
                    AvgPrec::new(i9, 90),
                    AvgPrec::new(i10, 95),
                ],
            }
        }
    }

    fn main() {
        println!("--------Test process_pr---------\n");
        let test_pr_pairs: [(f64, f64); 6] = [
            (0.5, 0.2),
            (0.6, 0.2),
            (0.4, 0.2),
            (0.9, 0.1),
            (0.95, 0.1),
            (0.7, 1.0),
        ];
        let mut vec_ret = process_pr(&test_pr_pairs);
        println!("{:?}", vec_ret);
        println!("------Test PR Interpolation-----\n");
        //let mut vec_test = vec![(0.5, 0.7), (0.3, 0.9), (0.4, 0.3)];
        let mut ap = pr_ip_cal_ap(&mut vec_ret, 50);
        println!("{:?}", ap);
        println!("--------Test cal_ap-------------\n");
        let m_aptable = ApTables::new(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1);
        let map_result = cal_map(&m_aptable);
        println!("The ap result is {}\n", map_result);
    }

    //input :[(precision, recall)], output Vec<(precision, recall)>, without duplicate recall
    //This function removes duplicate entries with same recall and stores the largest precision for each recall value
    fn process_pr(in_pairs: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let mut vec_pairs = Vec::<(f64, f64)>::new();
        let mut break_flag = false;
        for pair in in_pairs.iter() {
            if vec_pairs.len() == 0 {
                vec_pairs.push(*pair);
            } else {
                //Check if recall is already in vector
                break_flag = false;
                for i in 0..vec_pairs.len() {
                    if pair.1 == vec_pairs[i].1 {
                        if pair.0 > vec_pairs[i].0 {
                            vec_pairs[i].0 = pair.0;
                        }
                        break_flag = true;
                        break;
                    }
                }
                if break_flag == false {
                    vec_pairs.push(*pair);
                }
            }
        }
        return vec_pairs;
    }
    //input : Vec::<(precision, recall)>
    //This function smoothens the pr-curve and calculate the AP
    fn pr_ip_cal_ap(in_vec: &mut Vec<(f64, f64)>, iou: i32) -> AvgPrec {
        //sort the vector by recall
        in_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        //flatten the curve
        let mut cur_max_prec = in_vec.last().unwrap().0;
        //println!("{:?}", cur_max);
        for i in (0..in_vec.len()).rev() {
            if in_vec[i].0 > cur_max_prec {
                cur_max_prec = in_vec[i].0;
            }

            in_vec[i].0 = cur_max_prec;
        }
        //Interpolation
        let mut pr_table = PrTable::new(0.0, iou);
        let mut trav_vec = 0;
        println!("{:?}", in_vec);
        for i in 0..=100 {
            let f = i as f64 * 0.01;
            while trav_vec < in_vec.len() {
                //println!("{:?}", in_vec[trav_vec].1);
                if f <= in_vec[trav_vec].1 {
                    break;
                }
                trav_vec = trav_vec + 1;
            }
            if trav_vec == in_vec.len() {
                trav_vec = trav_vec - 1;
            }
            if f <= in_vec[trav_vec].1 {
                pr_table.pr_arr[i] = in_vec[trav_vec].0;
            }
        }
        let mut sum: f64 = 0.0;
        for i in 0..=100 {
            sum += pr_table.pr_arr[i];
        }
        sum = sum / 101.0;
        let ret_ap = AvgPrec::new(sum, iou);
        return ret_ap;
    }

    fn cal_map(ap_tables: &ApTables) -> f64 {
        let mut sum: f64 = 0.0;
        for ap in ap_tables.table_arr.iter() {
            sum = sum + ap.avg_prec;
        }
        sum = sum / 10.0;
        return sum;
    }
}
