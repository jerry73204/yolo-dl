use crate::{bbox::CyCxHW, common::*, ratio::Ratio, unit::Unit};

pub use area_tensor::*;
pub use cycxhw_tensor::*;
use into_tch_element::*;
pub use size_tensor::*;
pub use tlbr_tensor::*;

const EPSILON: f64 = 1e-16;

mod area_tensor {
    use super::*;

    /// Unchecked tensor of batched areas.
    #[derive(Debug, TensorLike)]
    pub struct AreaTensorUnchecked {
        pub area: Tensor,
    }

    /// Checked tensor of batched areas.
    #[derive(Debug, TensorLike, Getters)]
    pub struct AreaTensor {
        #[get = "pub"]
        pub(super) area: Tensor,
    }

    impl AreaTensor {
        pub fn num_samples(&self) -> i64 {
            let (num, _) = self.area.size2().unwrap();
            num
        }

        pub fn device(&self) -> Device {
            self.area.device()
        }
    }

    impl TryFrom<AreaTensorUnchecked> for AreaTensor {
        type Error = Error;

        fn try_from(from: AreaTensorUnchecked) -> Result<Self, Self::Error> {
            let AreaTensorUnchecked { area } = from;
            match area.size2()? {
                (_, 1) => (),
                _ => bail!("size_mismatch"),
            }
            Ok(Self { area })
        }
    }

    impl From<AreaTensor> for AreaTensorUnchecked {
        fn from(from: AreaTensor) -> Self {
            let AreaTensor { area } = from;
            Self { area }
        }
    }
}

mod size_tensor {
    use super::*;

    /// Unchecked tensor of batched sizes.
    #[derive(Debug, TensorLike)]
    pub struct SizeTensorUnchecked {
        pub h: Tensor,
        pub w: Tensor,
    }

    /// Checked tensor of batched sizes.
    #[derive(Debug, TensorLike, Getters)]
    pub struct SizeTensor {
        #[get = "pub"]
        pub(super) h: Tensor,
        #[get = "pub"]
        pub(super) w: Tensor,
    }

    impl SizeTensor {
        pub fn num_samples(&self) -> i64 {
            let (num, _) = self.h.size2().unwrap();
            num
        }

        pub fn device(&self) -> Device {
            self.h.device()
        }
    }

    impl TryFrom<SizeTensorUnchecked> for SizeTensor {
        type Error = Error;

        fn try_from(from: SizeTensorUnchecked) -> Result<Self, Self::Error> {
            let SizeTensorUnchecked { h, w } = from;
            match (h.size2()?, w.size2()?) {
                ((h_len, 1), (w_len, 1)) => ensure!(h_len == w_len, "size mismatch"),
                _ => bail!("size mismatch"),
            };
            ensure!(
                hashset! {
                    h.device(),
                    w.device(),
                }
                .len()
                    == 1,
                "device mismatch"
            );
            Ok(Self { h, w })
        }
    }

    impl From<SizeTensor> for SizeTensorUnchecked {
        fn from(from: SizeTensor) -> Self {
            let SizeTensor { h, w } = from;
            Self { h, w }
        }
    }
}

mod cycxhw_tensor {
    use super::*;

    /// Checked tensor of batched box parameters in CyCxHW format.
    #[derive(Debug, TensorLike, Getters)]
    pub struct CyCxHWTensor {
        #[get = "pub"]
        pub(crate) cy: Tensor,
        #[get = "pub"]
        pub(crate) cx: Tensor,
        #[get = "pub"]
        pub(crate) h: Tensor,
        #[get = "pub"]
        pub(crate) w: Tensor,
    }

    /// Unchecked tensor of batched box parameters in CyCxHW format.
    #[derive(Debug, TensorLike)]
    pub struct CyCxHWTensorUnchecked {
        pub cy: Tensor,
        pub cx: Tensor,
        pub h: Tensor,
        pub w: Tensor,
    }

    impl CyCxHWTensor {
        pub fn num_samples(&self) -> i64 {
            let (num, _) = self.cy.size2().unwrap();
            num
        }

        /// Compute box area.
        pub fn area(&self) -> AreaTensor {
            let Self { h, w, .. } = self;
            let area = h * w;
            AreaTensor { area }
        }

        /// Compute box size.
        pub fn size(&self) -> SizeTensor {
            let Self { h, w, .. } = self;
            SizeTensor {
                h: h.shallow_clone(),
                w: w.shallow_clone(),
            }
        }

        /// Compute the intersection area with the other box tensor.
        pub fn intersect_area_with(&self, other: &Self) -> AreaTensor {
            TLBRTensor::from(self).intersect_area_with(&TLBRTensor::from(other))
        }

        /// Compute the rectanble closure with the other box tensor.
        pub fn closure_with(&self, other: &Self) -> CyCxHWTensor {
            (&TLBRTensor::from(self).closure_with(&TLBRTensor::from(other))).into()
        }

        /// Compute the IoU score with the other box tensor.
        pub fn iou_with(&self, other: &Self) -> Tensor {
            let inter_area = self.intersect_area_with(other);
            let outer_area = self.area().area() + other.area().area() - inter_area.area() + EPSILON;
            let iou = inter_area.area() / outer_area;
            iou
        }

        /// Compute the GIoU score with the other box tensor.
        pub fn giou_with(&self, other: &Self) -> Tensor {
            let inter_area = self.intersect_area_with(other);
            let outer_area = self.area().area() + other.area().area() - inter_area.area() + EPSILON;
            let closure = self.closure_with(&other);
            let closure_area = closure.area();
            let iou = inter_area.area() / &outer_area;
            iou - (closure_area.area() - &outer_area) / (closure_area.area() + EPSILON)
        }

        /// Compute the DIoU score with the other box tensor.
        pub fn diou_with(&self, other: &Self) -> Tensor {
            let iou = self.iou_with(other);

            let closure = TLBRTensor::from(self).closure_with(&TLBRTensor::from(other));
            let closure_size = closure.size();

            let diagonal_square = closure_size.h().pow(2.0) + closure_size.w().pow(2.0) + EPSILON;
            let center_dist_square =
                (self.cy() - other.cy()).pow(2.0) + (self.cx() - other.cx()).pow(2.0);

            iou - center_dist_square / diagonal_square
        }

        /// Compute the CIoU score with the other box tensor.
        pub fn ciou_with(&self, other: &Self) -> Tensor {
            use std::f64::consts::PI;

            let iou = self.iou_with(other);

            let closure = TLBRTensor::from(self).closure_with(&TLBRTensor::from(other));
            let closure_size = closure.size();

            let pred_angle = self.h().atan2(&self.w());
            let target_angle = other.h().atan2(&other.w());

            let diagonal_square = closure_size.h().pow(2.0) + closure_size.w().pow(2.0) + EPSILON;
            let center_dist_square =
                (self.cy() - other.cy()).pow(2.0) + (self.cx() - other.cx()).pow(2.0);

            let shape_loss = (&pred_angle - &target_angle).pow(2.0) * 4.0 / PI.powi(2);
            let shape_loss_coef = tch::no_grad(|| &shape_loss / (1.0 - &iou + &shape_loss));

            iou - center_dist_square / diagonal_square + shape_loss_coef * shape_loss
        }

        /// Compute the Hausdorff distance with the other box tensor.
        pub fn hausdorff_distance_to(&self, other: &Self) -> Tensor {
            TLBRTensor::from(self).hausdorff_distance_to(&TLBRTensor::from(other))
        }
    }

    impl TryFrom<CyCxHWTensorUnchecked> for CyCxHWTensor {
        type Error = Error;

        fn try_from(from: CyCxHWTensorUnchecked) -> Result<Self, Self::Error> {
            let CyCxHWTensorUnchecked { cy, cx, h, w } = from;
            match (cy.size2()?, cx.size2()?, h.size2()?, w.size2()?) {
                ((cy_len, 1), (cx_len, 1), (h_len, 1), (w_len, 1)) => ensure!(
                    cy_len == cx_len && cy_len == h_len && cy_len == w_len,
                    "size mismatch"
                ),
                _ => bail!("size mismatch"),
            };
            ensure!(
                hashset! {
                    cy.device(),
                    cx.device(),
                    h.device(),
                    w.device(),
                }
                .len()
                    == 1,
                "device mismatch"
            );
            Ok(Self { cy, cx, h, w })
        }
    }

    impl From<CyCxHWTensor> for CyCxHWTensorUnchecked {
        fn from(from: CyCxHWTensor) -> Self {
            let CyCxHWTensor { cy, cx, h, w } = from;
            Self { cy, cx, h, w }
        }
    }

    impl From<&TLBRTensor> for CyCxHWTensor {
        fn from(from: &TLBRTensor) -> Self {
            let TLBRTensor { t, l, b, r } = from;
            let h = b - t;
            let w = r - l;
            let cy = t + &h / 2.0;
            let cx = l + &w / 2.0;
            Self { cy, cx, h, w }
        }
    }

    impl<U> From<&CyCxHWTensor> for Vec<CyCxHW<R64, U>>
    where
        U: Unit,
    {
        fn from(from: &CyCxHWTensor) -> Self {
            let bboxes: Vec<_> = izip!(
                Vec::<f32>::from(from.cy()),
                Vec::<f32>::from(from.cx()),
                Vec::<f32>::from(from.h()),
                Vec::<f32>::from(from.w()),
            )
            .map(|(cy, cx, h, w)| {
                CyCxHW::<R64, _>::from_cycxhw(
                    r64(cy as f64),
                    r64(cx as f64),
                    r64(h as f64),
                    r64(w as f64),
                )
                .unwrap()
            })
            .collect();
            bboxes
        }
    }

    impl<U> From<&CyCxHWTensor> for Vec<CyCxHW<f64, U>>
    where
        U: Unit,
    {
        fn from(from: &CyCxHWTensor) -> Self {
            let bboxes: Vec<_> = izip!(
                Vec::<f32>::from(from.cy()),
                Vec::<f32>::from(from.cx()),
                Vec::<f32>::from(from.h()),
                Vec::<f32>::from(from.w()),
            )
            .map(|(cy, cx, h, w)| {
                CyCxHW::<f64, _>::from_cycxhw(cy as f64, cx as f64, h as f64, w as f64).unwrap()
            })
            .collect();
            bboxes
        }
    }
}

mod tlbr_tensor {
    use super::*;

    /// Checked tensor of batched box parameters in TLBR format.
    #[derive(Debug, TensorLike, Getters)]
    pub struct TLBRTensor {
        #[get = "pub"]
        pub(crate) t: Tensor,
        #[get = "pub"]
        pub(crate) l: Tensor,
        #[get = "pub"]
        pub(crate) b: Tensor,
        #[get = "pub"]
        pub(crate) r: Tensor,
    }

    /// Unchecked tensor of batched box parameters in TLBR format.
    #[derive(Debug, TensorLike)]
    pub struct TLBRTensorUnchecked {
        pub t: Tensor,
        pub l: Tensor,
        pub b: Tensor,
        pub r: Tensor,
    }

    impl TLBRTensor {
        pub fn num_samples(&self) -> i64 {
            let (num, _) = self.t.size2().unwrap();
            num
        }

        pub fn device(&self) -> Device {
            self.t.device()
        }

        pub fn select(&self, index: i64) -> Self {
            let Self { t, l, b, r } = self;
            let range = index..(index + 1);
            Self {
                t: t.i((range.clone(), ..)),
                l: l.i((range.clone(), ..)),
                b: b.i((range.clone(), ..)),
                r: r.i((range, ..)),
            }
        }

        pub fn index_select(&self, indexes: &Tensor) -> Self {
            let Self { t, l, b, r } = self;
            let t = t.index_select(0, indexes);
            let l = l.index_select(0, indexes);
            let b = b.index_select(0, indexes);
            let r = r.index_select(0, indexes);
            Self { t, l, b, r }
        }

        /// Compute the box size.
        pub fn size(&self) -> SizeTensor {
            let Self { t, l, b, r } = self;
            let h = b - t;
            let w = r - l;

            SizeTensor { h, w }
        }

        /// Compute the box area.
        pub fn area(&self) -> AreaTensor {
            let SizeTensor { h, w } = self.size();
            let area = h * w;
            AreaTensor { area }
        }

        /// Compute the intersection area with the other box tensor.
        pub fn intersect_area_with(&self, other: &Self) -> AreaTensor {
            let Self {
                t: lhs_t,
                l: lhs_l,
                b: lhs_b,
                r: lhs_r,
            } = self;
            let Self {
                t: rhs_t,
                l: rhs_l,
                b: rhs_b,
                r: rhs_r,
            } = other;

            let max_t = lhs_t.max1(rhs_t);
            let max_l = lhs_l.max1(rhs_l);
            let min_b = lhs_b.min1(rhs_b);
            let min_r = lhs_r.min1(rhs_r);

            let inner_h = (min_b - max_t).clamp_min(0.0);
            let inner_w = (min_r - max_l).clamp_min(0.0);

            let area = inner_h * inner_w;

            AreaTensor { area }
        }

        /// Compute the rectangle closure with the other box tensor.
        pub fn closure_with(&self, other: &Self) -> Self {
            let Self {
                t: lhs_t,
                l: lhs_l,
                b: lhs_b,
                r: lhs_r,
            } = self;
            let Self {
                t: rhs_t,
                l: rhs_l,
                b: rhs_b,
                r: rhs_r,
            } = other;

            let min_t = lhs_t.min1(rhs_t);
            let min_l = lhs_l.min1(rhs_l);
            let max_b = lhs_b.max1(rhs_b);
            let max_r = lhs_r.max1(rhs_r);

            Self {
                t: min_t,
                l: min_l,
                b: max_b,
                r: max_r,
            }
        }

        /// Compute the Hausdorff distance with the other box tensor.
        pub fn hausdorff_distance_to(&self, other: &Self) -> Tensor {
            let TLBRTensor {
                t: tl,
                l: ll,
                b: bl,
                r: rl,
            } = self;
            let TLBRTensor {
                t: tr,
                l: lr,
                b: br,
                r: rr,
            } = other;

            let dt = tr - tl;
            let dl = lr - ll;
            let db = bl - br;
            let dr = rl - rr;

            let dt_l = dt.clamp_min(0.0);
            let dl_l = dl.clamp_min(0.0);
            let db_l = db.clamp_min(0.0);
            let dr_l = dr.clamp_min(0.0);

            let dt_r = (-&dt).clamp_min(0.0);
            let dl_r = (-&dl).clamp_min(0.0);
            let db_r = (-&db).clamp_min(0.0);
            let dr_r = (-&dr).clamp_min(0.0);

            (dt_l.pow(2.0) + dl_l.pow(2.0))
                .max1(&(dt_l.pow(2.0) + dr_l.pow(2.0)))
                .max1(&(db_l.pow(2.0) + dl_l.pow(2.0)))
                .max1(&(db_l.pow(2.0) + dr_l.pow(2.0)))
                .max1(&(dt_r.pow(2.0) + dl_r.pow(2.0)))
                .max1(&(dt_r.pow(2.0) + dr_r.pow(2.0)))
                .max1(&(db_r.pow(2.0) + dl_r.pow(2.0)))
                .max1(&(db_r.pow(2.0) + dr_r.pow(2.0)))
                .sqrt()
        }
    }

    impl TryFrom<TLBRTensorUnchecked> for TLBRTensor {
        type Error = Error;

        fn try_from(from: TLBRTensorUnchecked) -> Result<Self, Self::Error> {
            let TLBRTensorUnchecked { t, l, b, r } = from;
            match (t.size2()?, l.size2()?, b.size2()?, r.size2()?) {
                ((t_len, 1), (l_len, 1), (b_len, 1), (r_len, 1)) => ensure!(
                    t_len == l_len && t_len == b_len && t_len == r_len,
                    "size mismatch"
                ),
                _ => bail!("size mismatch"),
            };
            ensure!(
                hashset! {
                    t.device(),
                    l.device(),
                    b.device(),
                    r.device(),
                }
                .len()
                    == 1,
                "device mismatch"
            );
            Ok(Self { t, l, b, r })
        }
    }

    impl From<TLBRTensor> for TLBRTensorUnchecked {
        fn from(from: TLBRTensor) -> Self {
            let TLBRTensor { t, l, b, r } = from;
            Self { t, l, b, r }
        }
    }

    impl From<&CyCxHWTensor> for TLBRTensor {
        fn from(from: &CyCxHWTensor) -> Self {
            let CyCxHWTensor { cy, cx, h, w } = from;

            let t = cy - h / 2.0;
            let b = cy + h / 2.0;
            let l = cx - w / 2.0;
            let r = cx + w / 2.0;

            Self { t, l, b, r }
        }
    }

    impl<T, U> From<&[&CyCxHW<T, U>]> for TLBRTensor
    where
        T: Float + IntoTchElement,
        U: Unit,
    {
        fn from(from: &[&CyCxHW<T, U>]) -> Self {
            bboxes_to_tlbr_tensor(from)
        }
    }

    impl<T, U> From<&[CyCxHW<T, U>]> for TLBRTensor
    where
        T: Float + IntoTchElement,
        U: Unit,
    {
        fn from(from: &[CyCxHW<T, U>]) -> Self {
            bboxes_to_tlbr_tensor(from)
        }
    }

    impl<T, U> From<&Vec<CyCxHW<T, U>>> for TLBRTensor
    where
        T: Float + IntoTchElement,
        U: Unit,
    {
        fn from(from: &Vec<CyCxHW<T, U>>) -> Self {
            bboxes_to_tlbr_tensor(from.as_slice())
        }
    }

    impl<T, U> From<Vec<CyCxHW<T, U>>> for TLBRTensor
    where
        T: Float + IntoTchElement,
        U: Unit,
    {
        fn from(from: Vec<CyCxHW<T, U>>) -> Self {
            bboxes_to_tlbr_tensor(from.as_slice())
        }
    }

    impl<U> From<&TLBRTensor> for Vec<CyCxHW<R64, U>>
    where
        U: Unit,
    {
        fn from(from: &TLBRTensor) -> Self {
            (&CyCxHWTensor::from(from)).into()
        }
    }

    impl<U> From<&TLBRTensor> for Vec<CyCxHW<f64, U>>
    where
        U: Unit,
    {
        fn from(from: &TLBRTensor) -> Self {
            (&CyCxHWTensor::from(from)).into()
        }
    }

    fn bboxes_to_tlbr_tensor<B, T, U>(bboxes: &[B]) -> TLBRTensor
    where
        B: AsRef<CyCxHW<T, U>>,
        T: Float + IntoTchElement,
        U: Unit,
    {
        let (t_vec, l_vec, b_vec, r_vec) = bboxes
            .iter()
            .map(|bbox| {
                let tlbr = bbox.as_ref().to_tlbr();
                (
                    tlbr.t().into_tch_element(),
                    tlbr.l().into_tch_element(),
                    tlbr.b().into_tch_element(),
                    tlbr.r().into_tch_element(),
                )
            })
            .unzip_n_vec();

        let t_tensor = Tensor::of_slice(&t_vec);
        let l_tensor = Tensor::of_slice(&l_vec);
        let b_tensor = Tensor::of_slice(&b_vec);
        let r_tensor = Tensor::of_slice(&r_vec);

        TLBRTensor {
            t: t_tensor,
            l: l_tensor,
            b: b_tensor,
            r: r_tensor,
        }
    }
}

mod into_tch_element {
    use super::*;

    pub trait IntoTchElement
    where
        Self::Output: Element,
    {
        type Output;

        fn into_tch_element(self) -> Self::Output;
    }

    impl IntoTchElement for u8 {
        type Output = u8;

        fn into_tch_element(self) -> Self::Output {
            self
        }
    }

    impl IntoTchElement for i8 {
        type Output = i8;

        fn into_tch_element(self) -> Self::Output {
            self
        }
    }

    impl IntoTchElement for i16 {
        type Output = i16;

        fn into_tch_element(self) -> Self::Output {
            self
        }
    }

    impl IntoTchElement for bool {
        type Output = bool;

        fn into_tch_element(self) -> Self::Output {
            self
        }
    }

    impl IntoTchElement for f32 {
        type Output = f32;
        fn into_tch_element(self) -> Self::Output {
            self
        }
    }

    impl IntoTchElement for f64 {
        type Output = f64;
        fn into_tch_element(self) -> Self::Output {
            self
        }
    }

    impl IntoTchElement for R64 {
        type Output = f64;

        fn into_tch_element(self) -> Self::Output {
            self.raw()
        }
    }

    impl IntoTchElement for R32 {
        type Output = f32;

        fn into_tch_element(self) -> Self::Output {
            self.raw()
        }
    }

    impl IntoTchElement for Ratio {
        type Output = f64;

        fn into_tch_element(self) -> Self::Output {
            self.to_f64()
        }
    }
}
