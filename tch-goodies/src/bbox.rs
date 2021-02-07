//! Safe bounding box types and functions.

use crate::{
    common::*,
    ratio::Ratio,
    unit::{GridUnit, PixelUnit, RatioUnit, Unit},
};

pub use bbox::*;
pub use bbox_tensor::*;
use into_tch_element::*;
pub use labeled_bbox::*;

mod bbox_tensor {
    use super::*;

    #[derive(Debug, TensorLike, Getters)]
    pub struct SizeTensor {
        #[get = "pub"]
        h: Tensor,
        #[get = "pub"]
        w: Tensor,
    }

    #[derive(Debug, TensorLike, Getters)]
    pub struct AreaTensor {
        #[get = "pub"]
        area: Tensor,
    }

    #[derive(Debug, TensorLike, Getters)]
    pub struct CycxhwTensor {
        #[get = "pub"]
        cy: Tensor,
        #[get = "pub"]
        cx: Tensor,
        #[get = "pub"]
        h: Tensor,
        #[get = "pub"]
        w: Tensor,
    }

    #[derive(Debug, TensorLike, Getters)]
    pub struct TlbrTensor {
        #[get = "pub"]
        t: Tensor,
        #[get = "pub"]
        l: Tensor,
        #[get = "pub"]
        b: Tensor,
        #[get = "pub"]
        r: Tensor,
    }

    #[derive(Debug, TensorLike)]
    pub struct SizeTensorUnchecked {
        pub h: Tensor,
        pub w: Tensor,
    }

    #[derive(Debug, TensorLike)]
    pub struct AreaTensorUnchecked {
        pub area: Tensor,
    }

    #[derive(Debug, TensorLike)]
    pub struct CycxhwTensorUnchecked {
        pub cy: Tensor,
        pub cx: Tensor,
        pub h: Tensor,
        pub w: Tensor,
    }

    #[derive(Debug, TensorLike)]
    pub struct TlbrTensorUnchecked {
        pub t: Tensor,
        pub l: Tensor,
        pub b: Tensor,
        pub r: Tensor,
    }

    #[derive(Debug, TensorLike)]
    pub struct TlbrConfTensorUnchecked {
        pub tlbr: TlbrTensorUnchecked,
        pub conf: Tensor,
    }

    #[derive(Debug, TensorLike, Getters)]
    pub struct TlbrConfTensor {
        #[get = "pub"]
        tlbr: TlbrTensor,
        #[get = "pub"]
        conf: Tensor,
    }

    impl TlbrConfTensor {
        pub fn num_samples(&self) -> i64 {
            self.tlbr.num_samples()
        }

        pub fn index_select(&self, indexes: &Tensor) -> Self {
            let Self { tlbr, conf, .. } = self;
            let tlbr = tlbr.index_select(indexes);
            let conf = conf.index_select(0, indexes);
            Self { tlbr, conf }
        }

        pub fn device(&self) -> Device {
            self.tlbr.device()
        }
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

    impl AreaTensor {
        pub fn num_samples(&self) -> i64 {
            let (num, _) = self.area.size2().unwrap();
            num
        }

        pub fn device(&self) -> Device {
            self.area.device()
        }
    }

    impl CycxhwTensor {
        pub fn num_samples(&self) -> i64 {
            let (num, _) = self.cy.size2().unwrap();
            num
        }
    }

    impl TlbrTensor {
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

        pub fn size(&self) -> SizeTensor {
            let Self { t, l, b, r } = self;
            let h = b - t;
            let w = r - l;

            SizeTensor { h, w }
        }

        pub fn area(&self) -> AreaTensor {
            let SizeTensor { h, w } = self.size();
            let area = h * w;
            AreaTensor { area }
        }

        pub fn intersect_area(&self, other: &Self) -> AreaTensor {
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

        pub fn iou_with(&self, other: &Self) -> Tensor {
            let epsilon = 1e-4;
            let inter_area = self.intersect_area(other);
            let outer_area = self.area().area() + other.area().area() - inter_area.area() + epsilon;
            let iou = inter_area.area() / outer_area;
            iou
        }
    }

    impl TryFrom<TlbrConfTensorUnchecked> for TlbrConfTensor {
        type Error = Error;

        fn try_from(from: TlbrConfTensorUnchecked) -> Result<Self, Self::Error> {
            let TlbrConfTensorUnchecked { tlbr, conf } = from;
            let tlbr = TlbrTensor::try_from(tlbr)?;

            match conf.size2()? {
                (n_samples, 1) => ensure!(n_samples == tlbr.num_samples(), "size mismatch"),
                _ => bail!("size mismatch"),
            }
            ensure!(conf.device() == tlbr.device(), "device mismatch");

            Ok(Self { tlbr, conf })
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

    impl TryFrom<CycxhwTensorUnchecked> for CycxhwTensor {
        type Error = Error;

        fn try_from(from: CycxhwTensorUnchecked) -> Result<Self, Self::Error> {
            let CycxhwTensorUnchecked { cy, cx, h, w } = from;
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

    impl TryFrom<TlbrTensorUnchecked> for TlbrTensor {
        type Error = Error;

        fn try_from(from: TlbrTensorUnchecked) -> Result<Self, Self::Error> {
            let TlbrTensorUnchecked { t, l, b, r } = from;
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

    impl From<SizeTensor> for SizeTensorUnchecked {
        fn from(from: SizeTensor) -> Self {
            let SizeTensor { h, w } = from;
            Self { h, w }
        }
    }

    impl From<AreaTensor> for AreaTensorUnchecked {
        fn from(from: AreaTensor) -> Self {
            let AreaTensor { area } = from;
            Self { area }
        }
    }

    impl From<CycxhwTensor> for CycxhwTensorUnchecked {
        fn from(from: CycxhwTensor) -> Self {
            let CycxhwTensor { cy, cx, h, w } = from;
            Self { cy, cx, h, w }
        }
    }

    impl From<TlbrTensor> for TlbrTensorUnchecked {
        fn from(from: TlbrTensor) -> Self {
            let TlbrTensor { t, l, b, r } = from;
            Self { t, l, b, r }
        }
    }

    impl From<TlbrConfTensor> for TlbrConfTensorUnchecked {
        fn from(from: TlbrConfTensor) -> Self {
            let TlbrConfTensor { tlbr, conf } = from;
            Self {
                tlbr: tlbr.into(),
                conf,
            }
        }
    }

    impl From<&CycxhwTensor> for TlbrTensor {
        fn from(from: &CycxhwTensor) -> Self {
            let CycxhwTensor { cy, cx, h, w } = from;

            let t = cy - h / 2.0;
            let b = cy + h / 2.0;
            let l = cx - w / 2.0;
            let r = cx + w / 2.0;

            Self { t, l, b, r }
        }
    }

    impl<T, U> From<&[&BBox<T, U>]> for TlbrTensor
    where
        T: Num + Copy + IntoTchElement,
        U: Unit,
    {
        fn from(from: &[&BBox<T, U>]) -> Self {
            bboxes_to_tlbr_tensor(from)
        }
    }

    impl<T, U> From<&[BBox<T, U>]> for TlbrTensor
    where
        T: Num + Copy + IntoTchElement,
        U: Unit,
    {
        fn from(from: &[BBox<T, U>]) -> Self {
            bboxes_to_tlbr_tensor(from)
        }
    }

    impl<T, U> From<&Vec<BBox<T, U>>> for TlbrTensor
    where
        T: Num + Copy + IntoTchElement,
        U: Unit,
    {
        fn from(from: &Vec<BBox<T, U>>) -> Self {
            bboxes_to_tlbr_tensor(from.as_slice())
        }
    }

    impl<T, U> From<Vec<BBox<T, U>>> for TlbrTensor
    where
        T: Num + Copy + IntoTchElement,
        U: Unit,
    {
        fn from(from: Vec<BBox<T, U>>) -> Self {
            bboxes_to_tlbr_tensor(from.as_slice())
        }
    }

    impl<T, U> From<&[&LabeledBBox<T, U>]> for TlbrTensor
    where
        T: Num + Copy + IntoTchElement,
        U: Unit,
    {
        fn from(from: &[&LabeledBBox<T, U>]) -> Self {
            bboxes_to_tlbr_tensor(from)
        }
    }

    impl<T, U> From<&[LabeledBBox<T, U>]> for TlbrTensor
    where
        T: Num + Copy + IntoTchElement,
        U: Unit,
    {
        fn from(from: &[LabeledBBox<T, U>]) -> Self {
            bboxes_to_tlbr_tensor(from)
        }
    }

    impl<T, U> From<&Vec<LabeledBBox<T, U>>> for TlbrTensor
    where
        T: Num + Copy + IntoTchElement,
        U: Unit,
    {
        fn from(from: &Vec<LabeledBBox<T, U>>) -> Self {
            bboxes_to_tlbr_tensor(from.as_slice())
        }
    }

    impl<T, U> From<Vec<LabeledBBox<T, U>>> for TlbrTensor
    where
        T: Num + Copy + IntoTchElement,
        U: Unit,
    {
        fn from(from: Vec<LabeledBBox<T, U>>) -> Self {
            bboxes_to_tlbr_tensor(from.as_slice())
        }
    }

    fn bboxes_to_tlbr_tensor<B, T, U>(bboxes: &[B]) -> TlbrTensor
    where
        B: AsRef<BBox<T, U>>,
        T: Num + Copy + IntoTchElement,
        U: Unit,
    {
        let (t_vec, l_vec, b_vec, r_vec) = bboxes
            .iter()
            .map(|bbox| {
                let [t, l, b, r] = bbox.as_ref().tlbr();
                (
                    t.into_tch_element(),
                    l.into_tch_element(),
                    b.into_tch_element(),
                    r.into_tch_element(),
                )
            })
            .unzip_n_vec();

        let t_tensor = Tensor::of_slice(&t_vec);
        let l_tensor = Tensor::of_slice(&l_vec);
        let b_tensor = Tensor::of_slice(&b_vec);
        let r_tensor = Tensor::of_slice(&r_vec);

        TlbrTensor {
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

mod labeled_bbox {
    use super::*;

    // #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    // pub struct Corners<T, U>
    // where
    //     T: Num + Copy,
    //     U: Unit,
    // {
    //     pub tl: [T; 2],
    //     pub tr: [T; 2],
    //     pub bl: [T; 2],
    //     pub br: [T; 2],
    //     pub(super) _phantom: PhantomData<U>,
    // }

    /// Generic bounding box with an extra class ID.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct LabeledBBox<T, U>
    where
        T: Num + Copy,
        U: Unit,
    {
        pub bbox: BBox<T, U>,
        pub category_id: usize,
    }

    pub type LabeledPixelBBox<T> = LabeledBBox<T, PixelUnit>;
    pub type LabeledGridBBox<T> = LabeledBBox<T, GridUnit>;
    pub type LabeledRatioBBox = LabeledBBox<Ratio, RatioUnit>;

    impl<T, U> LabeledBBox<T, U>
    where
        T: Num + Copy,
        U: Unit,
    {
        pub fn cycxhw(&self) -> [T; 4] {
            self.bbox.cycxhw()
        }

        pub fn tlbr(&self) -> [T; 4]
        where
            T: Num + Copy,
        {
            self.bbox.tlbr()
        }

        pub fn map_elem<F, R>(&self, f: F) -> LabeledBBox<R, U>
        where
            F: FnMut(T) -> R,
            R: Num + Copy,
        {
            let Self {
                ref bbox,
                category_id,
            } = *self;

            LabeledBBox {
                bbox: bbox.map_elem(f),
                category_id,
            }
        }

        // pub fn corners(&self) -> Corners<T, U>
        // where
        //     T: Num,
        // {
        //     self.bbox.corners()
        // }

        pub fn to_unit<V>(&self, h_scale: T, w_scale: T) -> LabeledBBox<T, V>
        where
            V: Unit,
        {
            let Self {
                ref bbox,
                category_id,
            } = *self;

            LabeledBBox {
                bbox: bbox.to_unit(h_scale, w_scale),
                category_id,
            }
        }
    }

    impl<U> LabeledBBox<R64, U>
    where
        U: Unit,
    {
        pub fn scale(&self, scale: R64) -> Self {
            let Self {
                ref bbox,
                category_id,
            } = *self;
            Self {
                bbox: bbox.scale(scale),
                category_id,
            }
        }

        pub fn to_ratio_bbox(
            &self,
            image_height: usize,
            image_width: usize,
        ) -> Result<LabeledRatioBBox> {
            let bbox = self
                .bbox
                .to_ratio_bbox(R64::new(image_height as f64), R64::new(image_width as f64))?;
            let labeled_bbox = LabeledRatioBBox {
                bbox,
                category_id: self.category_id,
            };
            Ok(labeled_bbox)
        }
    }

    impl LabeledBBox<Ratio, RatioUnit> {
        pub fn scale(&self, scale: R64) -> Self {
            let Self {
                ref bbox,
                category_id,
            } = *self;
            Self {
                bbox: bbox.scale(scale),
                category_id,
            }
        }

        pub fn crop(&self, tlbr: [Ratio; 4]) -> Option<LabeledRatioBBox> {
            Some(LabeledRatioBBox {
                bbox: self.bbox.crop(tlbr)?,
                category_id: self.category_id,
            })
        }

        pub fn to_r64_bbox<U>(&self, height: usize, width: usize) -> LabeledBBox<R64, U>
        where
            U: Unit,
        {
            let Self {
                ref bbox,
                category_id,
            } = *self;
            LabeledBBox {
                bbox: bbox.to_r64_bbox(height, width),
                category_id,
            }
        }
    }

    impl<T, U> AsRef<BBox<T, U>> for LabeledBBox<T, U>
    where
        T: Num + Copy,
        U: Unit,
    {
        fn as_ref(&self) -> &BBox<T, U> {
            &self.bbox
        }
    }
}

mod bbox {
    use super::*;

    /// Bounding box in arbitrary units.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct BBox<T, U>
    where
        T: Num + Copy,
        U: Unit,
    {
        cycxhw: [T; 4],
        _phantom: PhantomData<U>,
    }

    pub type RatioBBox = BBox<Ratio, RatioUnit>;
    pub type GridBBox<T> = BBox<T, GridUnit>;
    pub type PixelBBox<T> = BBox<T, PixelUnit>;

    impl<T, U> BBox<T, U>
    where
        T: Num + Copy,
        U: Unit,
    {
        // pub fn corners(&self) -> Corners<T, U>
        // where
        //     T: Num,
        // {
        //     let [t, l, b, r] = self.tlbr();
        //     Corners {
        //         tl: [t, l],
        //         tr: [t, r],
        //         bl: [b, l],
        //         br: [b, r],
        //         _phantom: PhantomData,
        //     }
        // }

        pub fn tlbr(&self) -> [T; 4] {
            let two = T::one() + T::one();
            let [cy, cx, h, w] = self.cycxhw;
            let t = cy - h / two;
            let l = cx - w / two;
            let b = cy + h / two;
            let r = cx + w / two;
            [t, l, b, r]
        }

        pub fn cycxhw(&self) -> [T; 4] {
            self.cycxhw
        }

        pub fn map_elem<F, R>(&self, mut f: F) -> BBox<R, U>
        where
            F: FnMut(T) -> R,
            R: Num + Copy,
        {
            let [cy, cx, h, w] = self.cycxhw;
            BBox {
                cycxhw: [f(cy), f(cx), f(h), f(w)],
                _phantom: PhantomData,
            }
        }

        pub fn to_unit<V>(&self, h_scale: T, w_scale: T) -> BBox<T, V>
        where
            V: Unit,
        {
            let [cy, cx, h, w] = self.cycxhw;
            BBox {
                cycxhw: [cy * h_scale, cx * w_scale, h * h_scale, w * w_scale],
                _phantom: PhantomData,
            }
        }
    }

    impl<U> BBox<R64, U>
    where
        U: Unit,
    {
        pub fn try_from_tlbr(tlbr: [R64; 4]) -> Result<Self> {
            let [t, l, b, r] = tlbr;
            ensure!(t <= b && l <= r, "invalid tlbr {:?}", tlbr);

            let cy = (t + b) / 2.0;
            let cx = (l + r) / 2.0;
            let h = b - t;
            let w = r - l;

            Ok(Self {
                cycxhw: [cy, cx, h, w],
                _phantom: PhantomData,
            })
        }

        pub fn try_from_tlhw(tlhw: [R64; 4]) -> Result<Self> {
            let [t, l, h, w] = tlhw;
            ensure!(h >= 0.0 && w >= 0.0, "invalid tlhw {:?}", tlhw);
            let cy = t + h / 2.0;
            let cx = l + w / 2.0;

            Ok(Self {
                cycxhw: [cy, cx, h, w],
                _phantom: PhantomData,
            })
        }

        pub fn try_from_cycxhw(cycxhw: [R64; 4]) -> Result<Self> {
            let [_cy, _cx, h, w] = cycxhw;
            ensure!(h >= 0.0 && w >= 0.0, "invalid cycxhw {:?}", cycxhw);
            Ok(Self {
                cycxhw,
                _phantom: PhantomData,
            })
        }

        pub fn to_ratio_bbox(&self, max_height: R64, max_width: R64) -> Result<RatioBBox> {
            // construct ratio bbox
            let [orig_cy, orig_cx, orig_h, orig_w] = self.cycxhw;

            let ratio_cy = Ratio::try_from(orig_cy / max_height)?;
            let ratio_cx = Ratio::try_from(orig_cx / max_width)?;
            let ratio_h = Ratio::try_from(orig_h / max_height)?;
            let ratio_w = Ratio::try_from(orig_w / max_width)?;

            Ok(RatioBBox::try_from_cycxhw([
                ratio_cy, ratio_cx, ratio_h, ratio_w,
            ])?)
        }

        pub fn scale(&self, scale: R64) -> Self {
            let [cy, cx, h, w] = self.cycxhw;
            Self {
                cycxhw: [cy, cx, h * scale, w * scale],
                _phantom: PhantomData,
            }
        }
    }

    impl BBox<Ratio, RatioUnit> {
        pub fn try_from_cycxhw(cycxhw: [Ratio; 4]) -> Result<Self> {
            let [cy, cx, h, w] = cycxhw;

            // verify boundary
            let _ratio_t = cy.checked_sub(h / 2.0)?;
            let _ratio_l = cx.checked_sub(w / 2.0)?;
            let _ratio_b = cy.checked_add(h / 2.0)?;
            let _ratio_r = cx.checked_add(w / 2.0)?;

            Ok(Self {
                cycxhw: [cy, cx, h, w],
                _phantom: PhantomData,
            })
        }

        pub fn try_from_tlbr(tlbr: [Ratio; 4]) -> Result<Self> {
            let [t, l, b, r] = tlbr;
            let h = b.checked_sub(t)?;
            let w = r.checked_sub(l)?;
            let cy = t.checked_add(h / 2.0)?;
            let cx = l.checked_add(w / 2.0)?;
            Self::try_from_cycxhw([cy, cx, h, w])
        }

        pub fn crop(&self, tlbr: [Ratio; 4]) -> Option<RatioBBox> {
            let Self {
                cycxhw: [orig_cy, orig_cx, orig_h, orig_w],
                ..
            } = *self;

            let [margin_t, margin_l, margin_b, margin_r] = tlbr;

            let orig_t = orig_cy - orig_h / 2.0;
            let orig_l = orig_cx - orig_w / 2.0;
            let orig_b = orig_cy + orig_h / 2.0;
            let orig_r = orig_cx + orig_w / 2.0;

            let crop_t = orig_t.max(margin_t).min(margin_b);
            let crop_b = orig_b.max(margin_t).min(margin_b);
            let crop_l = orig_l.max(margin_l).min(margin_r);
            let crop_r = orig_r.max(margin_l).min(margin_r);

            if abs_diff_eq!(crop_t, crop_b) || abs_diff_eq!(crop_l, crop_r) {
                None
            } else {
                let crop_h = crop_b - crop_t;
                let crop_w = crop_r - crop_l;
                let crop_cy = crop_t + crop_h / 2.0;
                let crop_cx = crop_l + crop_w / 2.0;

                Some(RatioBBox {
                    cycxhw: [crop_cy, crop_cx, crop_h, crop_w],
                    _phantom: PhantomData,
                })
            }
        }

        pub fn to_r64_bbox<U>(&self, height: usize, width: usize) -> BBox<R64, U>
        where
            U: Unit,
        {
            let height = R64::new(height as f64);
            let width = R64::new(width as f64);
            let Self {
                cycxhw: [ratio_cy, ratio_cx, ratio_h, ratio_w],
                ..
            } = *self;

            let cy = ratio_cy.to_r64() * height;
            let cx = ratio_cx.to_r64() * width;
            let h = ratio_h.to_r64() * height;
            let w = ratio_w.to_r64() * width;

            BBox {
                cycxhw: [cy, cx, h, w],
                _phantom: PhantomData,
            }
        }

        pub fn scale(&self, scale: R64) -> Self {
            let Self {
                cycxhw: [orig_cy, orig_cx, orig_h, orig_w],
                ..
            } = *self;

            let tmp_h = orig_h.to_r64() * scale.max(R64::new(1.0));
            let tmp_w = orig_w.to_r64() * scale.max(R64::new(1.0));
            let new_t = (orig_cy.to_r64() - tmp_h / 2.0).max(R64::new(0.0));
            let new_b = (orig_cy.to_r64() + tmp_h / 2.0).min(R64::new(1.0));
            let new_l = (orig_cx.to_r64() - tmp_w / 2.0).max(R64::new(0.0));
            let new_r = (orig_cx.to_r64() + tmp_w / 2.0).min(R64::new(1.0));
            let new_cy = (new_t + new_b) / 2.0;
            let new_cx = (new_l + new_r) / 2.0;
            let new_h = new_b - new_t;
            let new_w = new_r - new_l;
            let cycxhw = [
                new_cy.try_into().unwrap(),
                new_cx.try_into().unwrap(),
                new_h.try_into().unwrap(),
                new_w.try_into().unwrap(),
            ];

            Self {
                cycxhw,
                _phantom: PhantomData,
            }
        }

        /// Compute intersection area in cycxhw format.
        pub fn intersect(&self, rhs: &Self) -> Option<[Ratio; 4]> {
            let [lt, ll, lb, lr] = self.tlbr();
            let [rt, rl, rb, rr] = rhs.tlbr();

            let t = lt.max(rt);
            let l = ll.max(rl);
            let b = lb.min(rb);
            let r = lr.min(rr);

            let h = b - t;
            let w = r - l;
            let cy = t + h / 2.0;
            let cx = l + w / 2.0;

            if abs_diff_eq!(cy, 0.0) || abs_diff_eq!(cx, 0.0) {
                return None;
            }

            Some([cy, cx, h, w])
        }
    }

    impl<T, U> AsRef<BBox<T, U>> for BBox<T, U>
    where
        T: Num + Copy,
        U: Unit,
    {
        fn as_ref(&self) -> &BBox<T, U> {
            &self
        }
    }
}
