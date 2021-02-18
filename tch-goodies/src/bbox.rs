//! Safe bounding box types and functions.

use crate::{
    common::*,
    ratio::Ratio,
    unit::{GridUnit, PixelUnit, RatioUnit, Unit, Unitless},
};

pub use cycxhw::*;
pub use labeled_cycxhw::*;
pub use tlbr::*;

mod tlbr {
    use super::*;

    /// Bounding box in arbitrary units.
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

    impl<T, U> TLBR<T, U>
    where
        T: Float,
        U: Unit,
    {
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

    /// Bounding box in arbitrary units.
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

    pub type RatioCyCxHW = CyCxHW<Ratio, RatioUnit>;
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

        pub fn to_unit<V>(&self) -> CyCxHW<T, V>
        where
            V: Unit,
        {
            CyCxHW {
                cy: self.cy,
                cx: self.cx,
                h: self.h,
                w: self.w,
                _phantom: PhantomData,
            }
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

        pub fn to_ratio_unit(&self, max_height: T, max_width: T) -> Result<RatioCyCxHW> {
            let (cy, cx, h, w) = move || -> Option<_> {
                let cy = <Ratio as NumCast>::from(self.cy / max_height)?;
                let cx = <Ratio as NumCast>::from(self.cx / max_width)?;
                let h = <Ratio as NumCast>::from(self.h / max_height)?;
                let w = <Ratio as NumCast>::from(self.w / max_width)?;
                Some((cy, cx, h, w))
            }()
            .ok_or_else(|| format_err!("range out of bound"))?;

            RatioCyCxHW::from_cycxhw(cy, cx, h, w)
        }
    }

    impl RatioCyCxHW {
        pub fn to_grid_unit<T>(&self, height: T, width: T) -> Result<GridCyCxHW<T>>
        where
            T: Float,
        {
            let zero = T::zero();
            ensure!(
                height >= zero && width >= zero,
                "height and width must be non-negative"
            );

            let Self { cy, cx, h, w, .. } = *self;
            let cy = <T as NumCast>::from(cy).unwrap();
            let cx = <T as NumCast>::from(cx).unwrap();
            let h = <T as NumCast>::from(h).unwrap();
            let w = <T as NumCast>::from(w).unwrap();
            Ok(GridCyCxHW {
                cy,
                cx,
                h,
                w,
                _phantom: PhantomData,
            })
        }

        pub fn to_pixel_unit<T>(&self, height: T, width: T) -> Result<PixelCyCxHW<T>>
        where
            T: Float,
        {
            let zero = T::zero();
            ensure!(
                height >= zero && width >= zero,
                "height and width must be non-negative"
            );

            let Self { cy, cx, h, w, .. } = *self;
            let cy = <T as NumCast>::from(cy).unwrap();
            let cx = <T as NumCast>::from(cx).unwrap();
            let h = <T as NumCast>::from(h).unwrap();
            let w = <T as NumCast>::from(w).unwrap();
            Ok(PixelCyCxHW {
                cy,
                cx,
                h,
                w,
                _phantom: PhantomData,
            })
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

mod labeled_cycxhw {
    use super::*;

    /// Generic bounding box with an extra class ID.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct LabeledCyCxHW<T, U>
    where
        T: Float,
        U: Unit,
    {
        pub bbox: CyCxHW<T, U>,
        pub category_id: usize,
    }

    pub type LabeledPixelCyCxHW<T> = LabeledCyCxHW<T, PixelUnit>;
    pub type LabeledGridCyCxHW<T> = LabeledCyCxHW<T, GridUnit>;
    pub type LabeledUnitlessCyCxHW<T> = LabeledCyCxHW<T, Unitless>;
    pub type LabeledRatioCyCxHW = LabeledCyCxHW<Ratio, RatioUnit>;

    impl<T, U> LabeledCyCxHW<T, U>
    where
        T: Float,
        U: Unit,
    {
        pub fn cy(&self) -> T {
            self.bbox.cy()
        }

        pub fn cx(&self) -> T {
            self.bbox.cx()
        }

        pub fn h(&self) -> T {
            self.bbox.h()
        }

        pub fn w(&self) -> T {
            self.bbox.w()
        }

        pub fn cycxhw(&self) -> CyCxHW<T, U>
        where
            CyCxHW<T, U>: Clone,
        {
            self.bbox.clone()
        }

        pub fn tlbr(&self) -> TLBR<T, U> {
            (&self.bbox).into()
        }

        pub fn cast<V>(&self) -> Option<LabeledCyCxHW<V, U>>
        where
            V: Float,
        {
            Some(LabeledCyCxHW {
                bbox: self.bbox.cast()?,
                category_id: self.category_id,
            })
        }

        /// Compute intersection area in TLBR format.
        pub fn intersect_with(&self, other: &CyCxHW<T, U>) -> Option<Self> {
            let intersection: CyCxHW<_, _> =
                (&TLBR::from(&self.bbox).intersect_with(&other.into())?).into();

            Some(Self {
                bbox: intersection,
                category_id: self.category_id,
            })
        }

        pub fn to_unit<V>(&self) -> LabeledCyCxHW<T, V>
        where
            V: Unit,
        {
            let Self {
                ref bbox,
                category_id,
            } = *self;

            LabeledCyCxHW {
                bbox: bbox.to_unit(),
                category_id,
            }
        }

        pub fn scale_size(&self, scale: T) -> Result<Self> {
            let Self {
                ref bbox,
                category_id,
            } = *self;
            Ok(Self {
                bbox: bbox.scale_size(scale)?,
                category_id,
            })
        }
    }

    impl LabeledRatioCyCxHW {
        pub fn to_grid_unit<T>(&self, height: T, width: T) -> Result<LabeledGridCyCxHW<T>>
        where
            T: Float,
        {
            Ok(LabeledGridCyCxHW {
                bbox: self.bbox.to_grid_unit(height, width)?,
                category_id: self.category_id,
            })
        }

        pub fn to_pixel_unit<T>(&self, height: T, width: T) -> Result<LabeledPixelCyCxHW<T>>
        where
            T: Float,
        {
            Ok(LabeledPixelCyCxHW {
                bbox: self.bbox.to_pixel_unit(height, width)?,
                category_id: self.category_id,
            })
        }
    }

    impl<T, U> AsRef<CyCxHW<T, U>> for LabeledCyCxHW<T, U>
    where
        T: Float,
        U: Unit,
    {
        fn as_ref(&self) -> &CyCxHW<T, U> {
            &self.bbox
        }
    }
}
