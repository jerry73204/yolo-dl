//! Safe bounding box types and functions.

use crate::{
    common::*,
    size::{GridSize, PixelSize, Size},
    unit::{GridUnit, PixelUnit, RatioUnit, Unit, Unitless},
};

pub use cycxhw::*;
pub use rect::*;
pub use rect_transform::*;
pub use tlbr::*;

const EPSILON: f64 = 1e-16;

mod rect_transform {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct RectTransform<T, U>
    where
        U: Unit,
    {
        sy: T,
        sx: T,
        ty: T,
        tx: T,
        _phantom: PhantomData<U>,
    }

    pub type PixelRectTransform<T> = RectTransform<T, PixelUnit>;
    pub type RatioRectTransform<T> = RectTransform<T, RatioUnit>;
    pub type GridRectTransform<T> = RectTransform<T, GridUnit>;

    impl<T, U> RectTransform<T, U>
    where
        U: Unit,
    {
        pub fn from_params(scale_y: T, scale_x: T, translate_y: T, translate_x: T) -> Self {
            Self {
                sy: scale_y,
                sx: scale_x,
                ty: translate_y,
                tx: translate_x,
                _phantom: PhantomData,
            }
        }

        pub fn from_rects(
            src: &impl Rect<Type = T, Unit = U>,
            tgt: &impl Rect<Type = T, Unit = U>,
        ) -> Self
        where
            T: Num + Copy,
        {
            let sy = tgt.h() / src.h();
            let sx = tgt.w() / src.w();
            let ty = tgt.t() - src.t() * sy;
            let tx = tgt.l() - src.l() * sx;

            Self {
                sy,
                sx,
                ty,
                tx,
                _phantom: PhantomData,
            }
        }

        pub fn from_resizing_exact(src_size: &Size<T, U>, tgt_size: &Size<T, U>) -> Self
        where
            T: Num + Copy + PartialOrd,
        {
            let src = TLBR::from_tlhw(T::zero(), T::zero(), src_size.h, src_size.w).unwrap();
            let tgt = TLBR::from_tlhw(T::zero(), T::zero(), tgt_size.h, tgt_size.w).unwrap();
            Self::from_rects(&src, &tgt)
        }

        pub fn from_resizing_letterbox(src_size: &Size<T, U>, tgt_size: &Size<T, U>) -> Self
        where
            T: Num + Copy + PartialOrd,
        {
            let (new_h, new_w) = if tgt_size.h * src_size.w <= tgt_size.w * src_size.h {
                let new_h = tgt_size.h;
                let new_w = src_size.w * tgt_size.h / src_size.h;
                (new_h, new_w)
            } else {
                let new_h = src_size.h * tgt_size.w / src_size.w;
                let new_w = tgt_size.w;
                (new_h, new_w)
            };

            let two = T::one() + T::one();
            let off_y = (tgt_size.h - new_h) / two;
            let off_x = (tgt_size.w - new_w) / two;

            let src = TLBR::from_tlhw(T::zero(), T::zero(), src_size.h, src_size.w).unwrap();
            let tgt = TLBR::from_tlhw(off_y, off_x, new_h, new_w).unwrap();

            Self::from_rects(&src, &tgt)
        }

        pub fn inverse(&self) -> Self
        where
            T: Float,
        {
            let sy = T::one() / self.sy;
            let sx = T::one() / self.sx;
            let ty = -self.ty / self.sy;
            let tx = -self.tx / self.sx;

            Self {
                sy,
                sx,
                ty,
                tx,
                _phantom: PhantomData,
            }
        }

        pub fn cast<V>(&self) -> Option<RectTransform<V, U>>
        where
            T: Copy + ToPrimitive,
            V: NumCast,
        {
            Some(RectTransform {
                sy: V::from(self.sy)?,
                sx: V::from(self.sx)?,
                ty: V::from(self.ty)?,
                tx: V::from(self.tx)?,
                _phantom: PhantomData,
            })
        }
    }

    impl<T, U> Mul<&TLBR<T, U>> for &RectTransform<T, U>
    where
        T: Num + Copy,
        U: Unit,
    {
        type Output = TLBR<T, U>;

        fn mul(self, rhs: &TLBR<T, U>) -> Self::Output {
            TLBR {
                t: rhs.t * self.sy + self.ty,
                l: rhs.l * self.sx + self.tx,
                b: rhs.b * self.sy + self.ty,
                r: rhs.r * self.sx + self.tx,
                _phantom: PhantomData,
            }
        }
    }

    impl<T, U> Mul<&CyCxHW<T, U>> for &RectTransform<T, U>
    where
        T: Num + Copy,
        U: Unit,
    {
        type Output = CyCxHW<T, U>;

        fn mul(self, rhs: &CyCxHW<T, U>) -> Self::Output {
            CyCxHW {
                cy: rhs.cy * self.sy + self.ty,
                cx: rhs.cx * self.sx + self.tx,
                h: rhs.h * self.sy,
                w: rhs.w * self.sx,
                _phantom: PhantomData,
            }
        }
    }

    impl<T, U> Mul<&RectTransform<T, U>> for &RectTransform<T, U>
    where
        T: Num + Copy,
        U: Unit,
    {
        type Output = RectTransform<T, U>;

        fn mul(self, rhs: &RectTransform<T, U>) -> Self::Output {
            RectTransform {
                sy: self.sy * rhs.sy,
                sx: self.sx * rhs.sx,
                ty: self.sy * rhs.ty + self.ty,
                tx: self.sx * rhs.tx + self.tx,
                _phantom: PhantomData,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn rect_transform_inverse() {
            let orig = PixelRectTransform::from_params(2.0, 2.0, 1.0, 1.0);
            assert_eq!(orig.inverse().inverse(), orig);
        }

        #[test]
        fn rect_resize_exact() {
            let transform = PixelRectTransform::from_resizing_exact(
                &PixelSize::from_hw(80.0, 80.0).unwrap(),
                &PixelSize::from_hw(20.0, 40.0).unwrap(),
            );
            let expect = PixelRectTransform::from_params(0.25, 0.5, 0.0, 0.0);
            assert_eq!(transform, expect);
        }

        #[test]
        fn rect_resize_letterbox() {
            let transform = PixelRectTransform::from_resizing_letterbox(
                &PixelSize::from_hw(80.0, 80.0).unwrap(),
                &PixelSize::from_hw(20.0, 40.0).unwrap(),
            );
            let expect = PixelRectTransform::from_params(0.25, 0.25, 0.0, 10.0);
            assert_eq!(transform, expect);
        }
    }
}

mod rect {
    use super::*;

    /// The generic rectangle.
    pub trait Rect
    where
        Self::Unit: Unit,
    {
        type Type;
        type Unit;

        fn t(&self) -> Self::Type;
        fn l(&self) -> Self::Type;
        fn b(&self) -> Self::Type;
        fn r(&self) -> Self::Type;
        fn cy(&self) -> Self::Type;
        fn cx(&self) -> Self::Type;
        fn h(&self) -> Self::Type;
        fn w(&self) -> Self::Type;

        fn area(&self) -> Self::Type
        where
            Self::Type: Num,
        {
            self.h() * self.w()
        }

        fn size(&self) -> Size<Self::Type, Self::Unit>
        where
            Self::Type: Num + PartialOrd,
        {
            Size::from_hw(self.h(), self.w()).unwrap()
        }
    }
}

mod tlbr {
    use super::*;

    /// Bounding box in TLBR format.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct TLBR<T, U>
    where
        U: Unit,
    {
        pub(super) t: T,
        pub(super) l: T,
        pub(super) b: T,
        pub(super) r: T,
        pub(super) _phantom: PhantomData<U>,
    }

    pub type RatioTLBR<T> = TLBR<T, RatioUnit>;
    pub type GridTLBR<T> = TLBR<T, GridUnit>;
    pub type PixelTLBR<T> = TLBR<T, PixelUnit>;
    pub type UnitlessTLBR<T> = TLBR<T, Unitless>;

    impl<T, U> TLBR<T, U>
    where
        U: Unit,
    {
        pub fn from_tlbr(t: T, l: T, b: T, r: T) -> Result<Self>
        where
            T: PartialOrd,
        {
            ensure!(b >= t && r >= l, "b >= t and r >= l must hold");

            Ok(Self {
                t,
                l,
                b,
                r,
                _phantom: PhantomData,
            })
        }

        pub fn from_tlhw(t: T, l: T, h: T, w: T) -> Result<Self>
        where
            T: Num + Copy + PartialOrd,
        {
            let b = t + h;
            let r = l + w;
            Self::from_tlbr(t, l, b, r)
        }

        pub fn tlbr_params(&self) -> [T; 4]
        where
            T: Copy,
        {
            [self.t, self.l, self.b, self.r]
        }

        pub fn to_cycxhw(&self) -> CyCxHW<T, U>
        where
            T: Num + Copy,
        {
            self.into()
        }

        /// Compute intersection area in TLBR format.
        pub fn intersect_with(&self, other: &Self) -> Option<Self>
        where
            T: Float,
        {
            let zero = T::zero();

            let t = self.t().max(other.t());
            let l = self.l().max(other.l());
            let b = self.b().min(other.b());
            let r = self.r().min(other.r());

            let h = b - t;
            let w = r - l;

            if h <= zero || w <= zero {
                return None;
            }

            Some(Self {
                t,
                l,
                b,
                r,
                _phantom: PhantomData,
            })
        }

        pub fn intersect_area_with(&self, other: &Self) -> T
        where
            T: Float,
        {
            self.intersect_with(other)
                .map(|size| size.area())
                .unwrap_or_else(T::zero)
        }

        /// Compute intersection area in TLBR format.
        pub fn closure_with(&self, other: &Self) -> Self
        where
            T: Float,
        {
            let t = self.t().min(other.t());
            let l = self.l().min(other.l());
            let b = self.b().max(other.b());
            let r = self.r().max(other.r());

            Self {
                t,
                l,
                b,
                r,
                _phantom: PhantomData,
            }
        }

        pub fn iou_with(&self, other: &Self) -> T
        where
            T: Float,
        {
            let inter_area = self.intersect_area_with(other);
            let union_area = self.area() + other.area() - inter_area + T::from(EPSILON).unwrap();
            inter_area / union_area
        }

        pub fn hausdorff_distance_to(&self, other: &Self) -> T
        where
            T: Float,
        {
            let zero = T::zero();
            let Self {
                t: tl,
                l: ll,
                b: bl,
                r: rl,
                ..
            } = *self;
            let Self {
                t: tr,
                l: lr,
                b: br,
                r: rr,
                ..
            } = *other;

            let dt = tr - tl;
            let dl = lr - ll;
            let db = bl - br;
            let dr = rl - rr;

            let dt_l = dt.max(zero);
            let dl_l = dl.max(zero);
            let db_l = db.max(zero);
            let dr_l = dr.max(zero);

            let dt_r = (-dt).max(zero);
            let dl_r = (-dl).max(zero);
            let db_r = (-db).max(zero);
            let dr_r = (-dr).max(zero);

            (dt_l.powi(2) + dl_l.powi(2))
                .max(dt_l.powi(2) + dr_l.powi(2))
                .max(db_l.powi(2) + dl_l.powi(2))
                .max(db_l.powi(2) + dr_l.powi(2))
                .max(dt_r.powi(2) + dl_r.powi(2))
                .max(dt_r.powi(2) + dr_r.powi(2))
                .max(db_r.powi(2) + dl_r.powi(2))
                .max(db_r.powi(2) + dr_r.powi(2))
                .sqrt()
        }

        pub fn cast<V>(&self) -> Option<TLBR<V, U>>
        where
            T: Copy + ToPrimitive,
            V: NumCast,
        {
            Some(TLBR {
                t: V::from(self.t)?,
                l: V::from(self.l)?,
                b: V::from(self.b)?,
                r: V::from(self.r)?,
                _phantom: PhantomData,
            })
        }
    }

    impl<T, U> Rect for TLBR<T, U>
    where
        T: Num + Copy,
        U: Unit,
    {
        type Type = T;
        type Unit = U;

        fn t(&self) -> Self::Type {
            self.t
        }

        fn l(&self) -> Self::Type {
            self.l
        }

        fn b(&self) -> Self::Type {
            self.b
        }

        fn r(&self) -> Self::Type {
            self.r
        }

        fn cy(&self) -> Self::Type {
            let one = T::one();
            let two = one + one;
            self.t + self.h() / two
        }

        fn cx(&self) -> Self::Type {
            let one = T::one();
            let two = one + one;
            self.l + self.w() / two
        }

        fn h(&self) -> Self::Type {
            self.b - self.t
        }

        fn w(&self) -> Self::Type {
            self.r - self.l
        }
    }

    impl<T, U> From<CyCxHW<T, U>> for TLBR<T, U>
    where
        T: Num + Copy,
        U: Unit,
    {
        fn from(from: CyCxHW<T, U>) -> Self {
            Self::from(&from)
        }
    }

    impl<T, U> From<&CyCxHW<T, U>> for TLBR<T, U>
    where
        T: Num + Copy,
        U: Unit,
    {
        fn from(from: &CyCxHW<T, U>) -> Self {
            let two = T::one() + T::one();
            let CyCxHW { cy, cx, h, w, .. } = *from;
            let t = cy - h / two;
            let l = cx - w / two;
            let b = cy + h / two;
            let r = cx + w / two;
            Self {
                t,
                l,
                b,
                r,
                _phantom: PhantomData,
            }
        }
    }
}

mod cycxhw {
    use super::*;

    /// Bounding box in CyCxHW format.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct CyCxHW<T, U>
    where
        U: Unit,
    {
        pub(super) cy: T,
        pub(super) cx: T,
        pub(super) h: T,
        pub(super) w: T,
        pub(super) _phantom: PhantomData<U>,
    }

    pub type RatioCyCxHW<T> = CyCxHW<T, RatioUnit>;
    pub type GridCyCxHW<T> = CyCxHW<T, GridUnit>;
    pub type PixelCyCxHW<T> = CyCxHW<T, PixelUnit>;
    pub type UnitlessCyCxHW<T> = CyCxHW<T, Unitless>;

    impl<T, U> CyCxHW<T, U>
    where
        U: Unit,
    {
        pub fn from_tlbr(t: T, l: T, b: T, r: T) -> Result<Self>
        where
            T: Num + Copy + PartialOrd,
        {
            let zero = T::zero();
            let two = T::one() + T::one();
            let cy = (t + b) / two;
            let cx = (l + r) / two;
            let h = b - t;
            let w = r - l;
            ensure!(
                h >= zero && w >= zero,
                "box height and width must be non-negative"
            );

            Ok(Self {
                cy,
                cx,
                h,
                w,
                _phantom: PhantomData,
            })
        }

        pub fn from_tlhw(t: T, l: T, h: T, w: T) -> Result<Self>
        where
            T: Num + Copy + PartialOrd,
        {
            let zero = T::zero();
            let two = T::one() + T::one();
            ensure!(
                h >= zero && w >= zero,
                "box height and width must be non-negative"
            );

            let cy = t + h / two;
            let cx = l + w / two;

            Ok(Self {
                cy,
                cx,
                h,
                w,
                _phantom: PhantomData,
            })
        }

        pub fn from_cycxhw(cy: T, cx: T, h: T, w: T) -> Result<Self>
        where
            T: Num + PartialOrd,
        {
            let zero = T::zero();
            ensure!(
                h >= zero && w >= zero,
                "box height and width must be non-negative"
            );

            Ok(Self {
                cy,
                cx,
                h,
                w,
                _phantom: PhantomData,
            })
        }

        pub fn cycxhw_params(&self) -> [T; 4]
        where
            T: Copy,
        {
            [self.cy, self.cx, self.h, self.w]
        }

        pub fn to_tlbr(&self) -> TLBR<T, U>
        where
            T: Num + Copy,
        {
            self.into()
        }

        pub fn cast<V>(&self) -> Option<CyCxHW<V, U>>
        where
            T: Copy + ToPrimitive,
            V: NumCast,
        {
            Some(CyCxHW {
                cy: V::from(self.cy)?,
                cx: V::from(self.cx)?,
                h: V::from(self.h)?,
                w: V::from(self.w)?,
                _phantom: PhantomData,
            })
        }

        pub fn scale_size(&self, scale: T) -> Result<Self>
        where
            T: Num + Copy + PartialOrd,
        {
            let Self { cy, cx, h, w, .. } = *self;
            let zero = T::zero();

            ensure!(scale >= zero, "scaling factor must be non-negative");
            let h = h * scale;
            let w = w * scale;
            debug_assert!(h >= zero && w >= zero);
            Ok(Self {
                cy,
                cx,
                h,
                w,
                _phantom: PhantomData,
            })
        }

        pub fn iou_with(&self, other: &Self) -> T
        where
            T: Float,
        {
            self.to_tlbr().iou_with(&other.to_tlbr())
        }

        pub fn hausdorff_distance_to(&self, other: &Self) -> T
        where
            T: Float,
        {
            self.to_tlbr().hausdorff_distance_to(&other.to_tlbr())
        }
    }

    impl<T, U> Rect for CyCxHW<T, U>
    where
        T: Copy + Num,
        U: Unit,
    {
        type Type = T;
        type Unit = U;

        fn t(&self) -> Self::Type {
            let two = T::one() + T::one();
            self.cy - self.h / two
        }

        fn l(&self) -> Self::Type {
            let two = T::one() + T::one();
            self.cx - self.w / two
        }

        fn b(&self) -> Self::Type {
            let two = T::one() + T::one();
            self.cy + self.h / two
        }

        fn r(&self) -> Self::Type {
            let two = T::one() + T::one();
            self.cx + self.w / two
        }

        fn cy(&self) -> Self::Type {
            self.cy
        }

        fn cx(&self) -> Self::Type {
            self.cx
        }

        fn h(&self) -> Self::Type {
            self.h
        }

        fn w(&self) -> Self::Type {
            self.w
        }
    }

    impl<T> PixelCyCxHW<T> {
        pub fn to_ratio_cycxhw(&self, size: &PixelSize<T>) -> RatioCyCxHW<T>
        where
            T: Num + Copy,
        {
            let cy = self.cy / size.h;
            let cx = self.cx / size.w;
            let h = self.h / size.h;
            let w = self.w / size.w;

            CyCxHW {
                cy,
                cx,
                h,
                w,
                _phantom: PhantomData,
            }
        }
    }

    impl<T> RatioCyCxHW<T> {
        pub fn to_pixel_cycxhw(&self, size: &PixelSize<T>) -> PixelCyCxHW<T>
        where
            T: Num + Copy,
        {
            let cy = self.cy * size.h;
            let cx = self.cx * size.w;
            let h = self.h * size.h;
            let w = self.w * size.w;

            CyCxHW {
                cy,
                cx,
                h,
                w,
                _phantom: PhantomData,
            }
        }

        pub fn to_grid_cycxhw(&self, size: &GridSize<T>) -> GridCyCxHW<T>
        where
            T: Num + Copy,
        {
            let cy = self.cy * size.h;
            let cx = self.cx * size.w;
            let h = self.h * size.h;
            let w = self.w * size.w;

            CyCxHW {
                cy,
                cx,
                h,
                w,
                _phantom: PhantomData,
            }
        }
    }

    impl<T, U> From<TLBR<T, U>> for CyCxHW<T, U>
    where
        T: Copy + Num,
        U: Unit,
    {
        fn from(from: TLBR<T, U>) -> Self {
            Self::from(&from)
        }
    }

    impl<T, U> From<&TLBR<T, U>> for CyCxHW<T, U>
    where
        T: Copy + Num,
        U: Unit,
    {
        fn from(from: &TLBR<T, U>) -> Self {
            let two = T::one() + T::one();
            let TLBR { t, l, b, r, .. } = *from;
            let h = b - t;
            let w = r - l;
            let cy = t + h / two;
            let cx = l + w / two;
            Self {
                cy,
                cx,
                h,
                w,
                _phantom: PhantomData,
            }
        }
    }

    impl<T, U> AsRef<CyCxHW<T, U>> for CyCxHW<T, U>
    where
        U: Unit,
    {
        fn as_ref(&self) -> &CyCxHW<T, U> {
            self
        }
    }
}

#[cfg(feature = "opencv")]
mod opencv_convert {
    use super::*;
    use opencv::core;

    impl<T> TryFrom<&core::Rect_<T>> for PixelTLBR<T>
    where
        T: Num + core::ValidRectType,
    {
        type Error = Error;

        fn try_from(from: &core::Rect_<T>) -> Result<Self, Self::Error> {
            let core::Rect_ {
                x: l,
                y: t,
                width: w,
                height: h,
            } = *from;
            Self::from_tlhw(t, l, h, w)
        }
    }

    impl<T> TryFrom<core::Rect_<T>> for PixelTLBR<T>
    where
        T: Num + core::ValidRectType,
    {
        type Error = Error;

        fn try_from(from: core::Rect_<T>) -> Result<Self, Self::Error> {
            (&from).try_into()
        }
    }

    impl<T> TryFrom<&core::Rect_<T>> for PixelCyCxHW<T>
    where
        T: Num + core::ValidRectType,
    {
        type Error = Error;

        fn try_from(from: &core::Rect_<T>) -> Result<Self, Self::Error> {
            let core::Rect_ {
                x: l,
                y: t,
                width: w,
                height: h,
            } = *from;
            Self::from_tlhw(t, l, h, w)
        }
    }

    impl<T> TryFrom<core::Rect_<T>> for PixelCyCxHW<T>
    where
        T: Num + core::ValidRectType,
    {
        type Error = Error;

        fn try_from(from: core::Rect_<T>) -> Result<Self, Self::Error> {
            (&from).try_into()
        }
    }

    impl<T> From<&PixelTLBR<T>> for core::Rect_<T>
    where
        T: Num + core::ValidRectType,
    {
        fn from(from: &PixelTLBR<T>) -> Self {
            Self {
                x: from.l(),
                y: from.t(),
                width: from.w(),
                height: from.h(),
            }
        }
    }

    impl<T> From<PixelTLBR<T>> for core::Rect_<T>
    where
        T: Num + core::ValidRectType,
    {
        fn from(from: PixelTLBR<T>) -> Self {
            (&from).into()
        }
    }

    impl<T> From<&PixelCyCxHW<T>> for core::Rect_<T>
    where
        T: Num + core::ValidRectType,
    {
        fn from(from: &PixelCyCxHW<T>) -> Self {
            Self {
                x: from.l(),
                y: from.t(),
                width: from.w(),
                height: from.h(),
            }
        }
    }

    impl<T> From<PixelCyCxHW<T>> for core::Rect_<T>
    where
        T: Num + core::ValidRectType,
    {
        fn from(from: PixelCyCxHW<T>) -> Self {
            (&from).into()
        }
    }
}
