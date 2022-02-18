use super::{Rect, TLBR};
use crate::{common::*, Transform};

/// Bounding box in CyCxHW format.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CyCxHW<T> {
    pub(crate) cy: T,
    pub(crate) cx: T,
    pub(crate) h: T,
    pub(crate) w: T,
}

impl<T> CyCxHW<T> {
    pub fn try_cast<V>(self) -> Option<CyCxHW<V>>
    where
        T: ToPrimitive,
        V: NumCast,
    {
        Some(CyCxHW {
            cy: V::from(self.cy)?,
            cx: V::from(self.cx)?,
            h: V::from(self.h)?,
            w: V::from(self.w)?,
        })
    }

    pub fn cast<V>(self) -> CyCxHW<V>
    where
        T: ToPrimitive,
        V: NumCast,
    {
        self.try_cast().unwrap()
    }
}

impl<T> CyCxHW<T>
where
    T: Copy + Num,
{
    pub fn transform(&self, transform: &Transform<T>) -> Self {
        CyCxHW {
            cy: self.cy * transform.sy + transform.ty,
            cx: self.cx * transform.sx + transform.tx,
            h: self.h * transform.sy,
            w: self.w * transform.sx,
        }
    }
}

impl<T> CyCxHW<T>
where
    T: Copy + Num + PartialOrd,
{
    pub fn try_scale(&self, scale: T) -> Result<Self> {
        let zero = T::zero();
        ensure!(scale > zero, "scaling factor must be positive");

        let Self { cy, cx, h, w, .. } = *self;

        let h = h * scale;
        let w = w * scale;
        debug_assert!(h >= zero && w >= zero);
        Ok(Self { cy, cx, h, w })
    }

    pub fn scale(&self, scale: T) -> Self {
        self.try_scale(scale).unwrap()
    }

    pub fn try_scale_hw(&self, scale_h: T, scale_w: T) -> Result<Self> {
        let zero = T::zero();
        ensure!(
            scale_h > zero && scale_w > zero,
            "scaling factor must be positive"
        );

        let Self { cy, cx, h, w, .. } = *self;
        let h = h * scale_h;
        let w = w * scale_w;
        debug_assert!(h >= zero && w >= zero);
        Ok(Self { cy, cx, h, w })
    }

    pub fn scale_hw(&self, scale_h: T, scale_w: T) -> Self {
        self.try_scale_hw(scale_h, scale_w).unwrap()
    }
}

impl<T> Rect for CyCxHW<T>
where
    T: Copy + Num + PartialOrd,
{
    type Type = T;

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

    fn try_from_tlbr(tlbr: [T; 4]) -> Result<Self> {
        let [t, l, b, r] = tlbr;
        let zero = T::zero();
        let two = T::one() + T::one();
        let h = b - t;
        let w = r - l;
        let cy = t + h / two;
        let cx = l + w / two;
        ensure!(
            h >= zero && w >= zero,
            "box height and width must be non-negative"
        );

        Ok(Self { cy, cx, h, w })
    }

    fn try_from_tlhw(tlhw: [T; 4]) -> Result<Self> {
        let [t, l, h, w] = tlhw;
        let zero = T::zero();
        let two = T::one() + T::one();
        ensure!(
            h >= zero && w >= zero,
            "box height and width must be non-negative"
        );

        let cy = t + h / two;
        let cx = l + w / two;

        Ok(Self { cy, cx, h, w })
    }

    fn try_from_cycxhw(cycxhw: [T; 4]) -> Result<Self> {
        let [cy, cx, h, w] = cycxhw;
        let zero = T::zero();
        ensure!(
            h >= zero && w >= zero,
            "box height and width must be non-negative"
        );

        Ok(Self { cy, cx, h, w })
    }
}

impl<T> From<TLBR<T>> for CyCxHW<T>
where
    T: Copy + Num,
{
    fn from(from: TLBR<T>) -> Self {
        Self::from(&from)
    }
}

impl<T> From<&TLBR<T>> for CyCxHW<T>
where
    T: Copy + Num,
{
    fn from(from: &TLBR<T>) -> Self {
        let two = T::one() + T::one();
        let TLBR { t, l, b, r, .. } = *from;
        let h = b - t;
        let w = r - l;
        let cy = t + h / two;
        let cx = l + w / two;
        Self { cy, cx, h, w }
    }
}
