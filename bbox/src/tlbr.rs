use super::{CyCxHW, Rect};
use crate::{common::*, RectElement};

/// Bounding box in TLBR format.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TLBR<T> {
    pub(crate) t: T,
    pub(crate) l: T,
    pub(crate) b: T,
    pub(crate) r: T,
}

impl<T> TLBR<T> {
    pub fn cast<V>(&self) -> Option<TLBR<V>>
    where
        T: Copy + ToPrimitive,
        V: NumCast,
    {
        Some(TLBR {
            t: V::from(self.t)?,
            l: V::from(self.l)?,
            b: V::from(self.b)?,
            r: V::from(self.r)?,
        })
    }
}

impl<T> Rect for TLBR<T>
where
    T: RectElement,
{
    type Type = T;

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

    fn try_from_cycxhw(cycxhw: [Self::Type; 4]) -> Result<Self>
    where
        T: Num + Copy + PartialOrd,
    {
        let [cy, cx, h, w] = cycxhw;
        let zero = T::zero();
        ensure!(h >= zero && w >= zero, "h and w must be non-negative");

        let two = T::one() + T::one();
        let t = cy - h / two;
        let b = cy + h / two;
        let l = cx - w / two;
        let r = cx + w / two;

        Ok(Self { t, l, b, r })
    }

    fn try_from_tlbr(tlbr: [Self::Type; 4]) -> Result<Self>
    where
        T: PartialOrd,
    {
        let [t, l, b, r] = tlbr;
        ensure!(b >= t && r >= l, "b >= t and r >= l must hold");

        Ok(Self { t, l, b, r })
    }

    fn try_from_tlhw(tlhw: [Self::Type; 4]) -> Result<Self>
    where
        T: Num + Copy + PartialOrd,
    {
        let [t, l, h, w] = tlhw;
        let b = t + h;
        let r = l + w;
        Self::try_from_tlbr([t, l, b, r])
    }
}

impl<T> From<CyCxHW<T>> for TLBR<T>
where
    T: Num + Copy,
{
    fn from(from: CyCxHW<T>) -> Self {
        Self::from(&from)
    }
}

impl<T> From<&CyCxHW<T>> for TLBR<T>
where
    T: Num + Copy,
{
    fn from(from: &CyCxHW<T>) -> Self {
        let two = T::one() + T::one();
        let CyCxHW { cy, cx, h, w, .. } = *from;
        let t = cy - h / two;
        let l = cx - w / two;
        let b = cy + h / two;
        let r = cx + w / two;
        Self { t, l, b, r }
    }
}
