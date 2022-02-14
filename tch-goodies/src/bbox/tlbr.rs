use super::{CyCxHW, Rect};
use crate::{
    common::*,
    unit::{GridUnit, PixelUnit, RatioUnit, Unit, Unitless},
};

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
    pub fn from_cycxhw(cy: T, cx: T, h: T, w: T) -> Result<Self>
    where
        T: Num + Copy + PartialOrd,
    {
        let zero = T::zero();
        ensure!(h >= zero && w >= zero, "h and w must be non-negative");

        let two = T::one() + T::one();
        let t = cy - h / two;
        let b = cy + h / two;
        let l = cx - w / two;
        let r = cx + w / two;

        Ok(Self {
            t,
            l,
            b,
            r,
            _phantom: PhantomData,
        })
    }

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
