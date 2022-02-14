use super::{Rect, TLBR};
use crate::{
    common::*,
    size::{GridSize, PixelSize},
    unit::{GridUnit, PixelUnit, RatioUnit, Unit, Unitless},
};

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
