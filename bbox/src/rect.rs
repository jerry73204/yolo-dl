use super::{CyCxHW, TLBR};
use crate::common::*;

/// The generic rectangle.
pub trait Rect {
    type Type;

    fn t(&self) -> Self::Type;
    fn l(&self) -> Self::Type;
    fn b(&self) -> Self::Type;
    fn r(&self) -> Self::Type;
    fn cy(&self) -> Self::Type;
    fn cx(&self) -> Self::Type;
    fn h(&self) -> Self::Type;
    fn w(&self) -> Self::Type;

    fn try_from_tlbr(tlbr: [Self::Type; 4]) -> Result<Self>
    where
        Self: Sized;

    fn try_from_tlhw(tlhw: [Self::Type; 4]) -> Result<Self>
    where
        Self: Sized;

    fn try_from_cycxhw(cycxhw: [Self::Type; 4]) -> Result<Self>
    where
        Self: Sized;
}

pub trait RectNum: Rect
where
    Self::Type: Num + PartialOrd,
{
    fn from_tlbr(tlbr: [Self::Type; 4]) -> Self
    where
        Self: Sized,
    {
        Self::try_from_tlbr(tlbr).unwrap()
    }

    fn from_tlhw(tlhw: [Self::Type; 4]) -> Self
    where
        Self: Sized,
    {
        Self::try_from_tlhw(tlhw).unwrap()
    }

    fn from_cycxhw(cycxhw: [Self::Type; 4]) -> Self
    where
        Self: Sized,
    {
        Self::try_from_cycxhw(cycxhw).unwrap()
    }

    fn cycxhw(&self) -> [Self::Type; 4] {
        [self.cy(), self.cx(), self.h(), self.w()]
    }

    fn tlbr(&self) -> [Self::Type; 4] {
        [self.t(), self.l(), self.b(), self.r()]
    }

    fn tlhw(&self) -> [Self::Type; 4] {
        [self.t(), self.l(), self.h(), self.w()]
    }

    fn hw(&self) -> [Self::Type; 2] {
        [self.h(), self.w()]
    }

    fn to_cycxhw(&self) -> CyCxHW<Self::Type> {
        CyCxHW {
            cy: self.cy(),
            cx: self.cx(),
            h: self.h(),
            w: self.w(),
        }
    }

    fn to_tlbr(&self) -> TLBR<Self::Type> {
        TLBR {
            t: self.t(),
            l: self.l(),
            b: self.b(),
            r: self.r(),
        }
    }

    fn area(&self) -> <Self::Type as Mul<Self::Type>>::Output
    where
        Self::Type: Mul<Self::Type>,
    {
        self.h() * self.w()
    }
}

pub trait RectFloat: RectNum
where
    Self::Type: Float,
{
    /// Compute intersection area in TLBR format.
    fn closure_with<R>(&self, other: &R) -> TLBR<Self::Type>
    where
        R: Rect<Type = Self::Type>,
    {
        let t = self.t().min(other.t());
        let l = self.l().min(other.l());
        let b = self.b().max(other.b());
        let r = self.r().max(other.r());
        TLBR::from_tlbr([t, l, b, r])
    }

    fn intersect_with<R>(&self, other: &R) -> Option<TLBR<Self::Type>>
    where
        R: Rect<Type = Self::Type>,
    {
        let t = self.t().max(other.t());
        let l = self.l().max(other.l());
        let b = self.b().min(other.b());
        let r = self.r().min(other.r());
        (b > t && r > l).then(|| TLBR::from_tlbr([t, l, b, r]))
    }

    fn intersection_area_with<R>(&self, other: &R) -> Self::Type
    where
        R: Rect<Type = Self::Type>,
    {
        self.intersect_with(other)
            .map(|rect| rect.area())
            .unwrap_or_else(Self::Type::zero)
    }

    fn iou_with<R>(&self, other: &R, epsilon: Self::Type) -> Self::Type
    where
        R: Rect<Type = Self::Type>,
    {
        let inter_area = self.intersection_area_with(other);
        let union_area = self.area() + other.area() - inter_area + epsilon;
        inter_area / union_area
    }

    fn hausdorff_distance_to<R>(&self, other: &R) -> Self::Type
    where
        R: Rect<Type = Self::Type>,
    {
        let zero = Self::Type::zero();
        let [tl, ll, bl, rl] = self.tlbr();
        let [tr, lr, br, rr] = other.tlbr();

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
}

impl<T> RectNum for T
where
    T: Rect,
    T::Type: Num + PartialOrd,
{
}

impl<T> RectFloat for T
where
    T: Rect,
    T::Type: Float,
{
}
