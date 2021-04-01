//! Safe bounding box types and functions.

use crate::{
    common::*,
    size::Size,
    unit::{GridUnit, PixelUnit, RatioUnit, Unit, Unitless},
};
use num_traits::NumCast;

pub use cycxhw::*;
pub use tlbr::*;

const EPSILON: f64 = 1e-16;

mod tlbr {
    use super::*;

    /// Bounding box in TLBR format.
    #[derive(Debug, Clone, PartialEq, Eq, Hash, CopyGetters)]
    pub struct TLBR<T, U>
    where
        T: Float,
        U: Unit,
    {
        #[get_copy = "pub"]
        pub(super) t: T,
        #[get_copy = "pub"]
        pub(super) l: T,
        #[get_copy = "pub"]
        pub(super) b: T,
        #[get_copy = "pub"]
        pub(super) r: T,
        _phantom: PhantomData<U>,
    }

    pub type RatioTLBR<T> = TLBR<T, RatioUnit>;
    pub type GridTLBR<T> = TLBR<T, GridUnit>;
    pub type PixelTLBR<T> = TLBR<T, PixelUnit>;
    pub type UnitlessTLBR<T> = TLBR<T, Unitless>;

    impl<T, U> TLBR<T, U>
    where
        T: Float,
        U: Unit,
    {
        pub fn from_tlbr(t: T, l: T, b: T, r: T) -> Result<Self> {
            ensure!(b >= t && r >= l, "b >= t and r >= l must hold");

            Ok(Self {
                t,
                l,
                b,
                r,
                _phantom: PhantomData,
            })
        }

        pub fn tlbr_params(&self) -> [T; 4] {
            [self.t, self.l, self.b, self.r]
        }

        pub fn size(&self) -> Size<T, U> {
            let Self { t, l, b, r, .. } = *self;
            let h = b - t;
            let w = r - l;
            Size::new(h, w).unwrap()
        }

        pub fn to_cycxhw(&self) -> CyCxHW<T, U> {
            self.into()
        }

        pub fn area(&self) -> T {
            self.size().area()
        }

        /// Compute intersection area in TLBR format.
        pub fn intersect_with(&self, other: &Self) -> Option<Self> {
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

        pub fn intersect_area_with(&self, other: &Self) -> T {
            self.intersect_with(other)
                .map(|size| size.area())
                .unwrap_or_else(T::zero)
        }

        /// Compute intersection area in TLBR format.
        pub fn closure_with(&self, other: &Self) -> Self {
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

        pub fn iou_with(&self, other: &Self) -> T {
            let inter_area = self.intersect_area_with(other);
            let union_area = self.area() + other.area() - inter_area + T::from(EPSILON).unwrap();
            inter_area / union_area
        }

        pub fn hausdorff_distance_to(&self, other: &Self) -> T {
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
            V: Float,
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

    impl<T, U> From<CyCxHW<T, U>> for TLBR<T, U>
    where
        T: Float,
        U: Unit,
    {
        fn from(from: CyCxHW<T, U>) -> Self {
            Self::from(&from)
        }
    }

    impl<T, U> From<&CyCxHW<T, U>> for TLBR<T, U>
    where
        T: Float,
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
    #[derive(Debug, Clone, PartialEq, Eq, Hash, CopyGetters)]
    pub struct CyCxHW<T, U>
    where
        T: Float,
        U: Unit,
    {
        #[get_copy = "pub"]
        pub(super) cy: T,
        #[get_copy = "pub"]
        pub(super) cx: T,
        #[get_copy = "pub"]
        pub(super) h: T,
        #[get_copy = "pub"]
        pub(super) w: T,
        _phantom: PhantomData<U>,
    }

    pub type RatioCyCxHW<T> = CyCxHW<T, RatioUnit>;
    pub type GridCyCxHW<T> = CyCxHW<T, GridUnit>;
    pub type PixelCyCxHW<T> = CyCxHW<T, PixelUnit>;
    pub type UnitlessCyCxHW<T> = CyCxHW<T, Unitless>;

    impl<T, U> CyCxHW<T, U>
    where
        T: Float,
        U: Unit,
    {
        pub fn from_tlbr(t: T, l: T, b: T, r: T) -> Result<Self> {
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

        pub fn from_tlhw(t: T, l: T, h: T, w: T) -> Result<Self> {
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

        pub fn from_cycxhw(cy: T, cx: T, h: T, w: T) -> Result<Self> {
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

        pub fn cycxhw_params(&self) -> [T; 4] {
            [self.cy, self.cx, self.h, self.w]
        }

        pub fn size(&self) -> Size<T, U> {
            let Self { h, w, .. } = *self;
            Size::new(h, w).unwrap()
        }

        pub fn area(&self) -> T {
            let Self { h, w, .. } = *self;
            h * w
        }

        pub fn to_tlbr(&self) -> TLBR<T, U> {
            self.into()
        }

        pub fn cast<V>(&self) -> Option<CyCxHW<V, U>>
        where
            V: Float,
        {
            Some(CyCxHW {
                cy: V::from(self.cy)?,
                cx: V::from(self.cx)?,
                h: V::from(self.h)?,
                w: V::from(self.w)?,
                _phantom: PhantomData,
            })
        }

        pub fn scale_to_unit<S, V>(&self, h_scale: S, w_scale: S) -> Result<CyCxHW<S, V>>
        where
            S: Float,
            V: Unit,
        {
            let zero = S::zero();
            ensure!(
                h_scale >= zero && w_scale >= zero,
                "height and width must be non-negative"
            );

            let Self { cy, cx, h, w, .. } = *self;
            let cy = <S as NumCast>::from(cy).unwrap() * h_scale;
            let cx = <S as NumCast>::from(cx).unwrap() * w_scale;
            let h = <S as NumCast>::from(h).unwrap() * h_scale;
            let w = <S as NumCast>::from(w).unwrap() * w_scale;

            Ok(CyCxHW {
                cy,
                cx,
                h,
                w,
                _phantom: PhantomData,
            })
        }

        pub fn scale_size(&self, scale: T) -> Result<Self> {
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

        pub fn iou_with(&self, other: &Self) -> T {
            self.to_tlbr().iou_with(&other.to_tlbr())
        }

        pub fn hausdorff_distance_to(&self, other: &Self) -> T {
            self.to_tlbr().hausdorff_distance_to(&other.to_tlbr())
        }
    }

    impl<T, U> From<TLBR<T, U>> for CyCxHW<T, U>
    where
        T: Float,
        U: Unit,
    {
        fn from(from: TLBR<T, U>) -> Self {
            Self::from(&from)
        }
    }

    impl<T, U> From<&TLBR<T, U>> for CyCxHW<T, U>
    where
        T: Float,
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
        T: Float,
        U: Unit,
    {
        fn as_ref(&self) -> &CyCxHW<T, U> {
            &self
        }
    }
}
