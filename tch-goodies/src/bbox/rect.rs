use super::{CyCxHW, EPSILON, TLBR};
use crate::{common::*, unit::Unit, Size};

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

    fn cycxhw(&self) -> [Self::Type; 4] {
        [self.cy(), self.cx(), self.h(), self.w()]
    }

    fn tlbr(&self) -> [Self::Type; 4] {
        [self.t(), self.l(), self.b(), self.r()]
    }

    fn tlhw(&self) -> [Self::Type; 4] {
        [self.t(), self.l(), self.h(), self.w()]
    }

    fn to_cycxhw(&self) -> CyCxHW<Self::Type, Self::Unit> {
        CyCxHW {
            cy: self.cy(),
            cx: self.cx(),
            h: self.h(),
            w: self.w(),
            _phantom: PhantomData,
        }
    }

    fn to_tlbr(&self) -> TLBR<Self::Type, Self::Unit> {
        TLBR {
            t: self.t(),
            l: self.l(),
            b: self.b(),
            r: self.r(),
            _phantom: PhantomData,
        }
    }

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

    /// Compute intersection area in TLBR format.
    fn closure_with<R>(&self, other: &R) -> TLBR<Self::Type, Self::Unit>
    where
        Self::Type: Float,
        R: Rect<Type = Self::Type, Unit = Self::Unit>,
    {
        let t = self.t().min(other.t());
        let l = self.l().min(other.l());
        let b = self.b().max(other.b());
        let r = self.r().max(other.r());
        TLBR::from_tlbr(t, l, b, r).unwrap()
    }

    fn intersect_with<R>(&self, other: &R) -> Option<TLBR<Self::Type, Self::Unit>>
    where
        Self::Type: Float,
        R: Rect<Type = Self::Type, Unit = Self::Unit>,
    {
        let zero = Self::Type::zero();

        let t = self.t().max(other.t());
        let l = self.l().max(other.l());
        let b = self.b().min(other.b());
        let r = self.r().min(other.r());

        let h = b - t;
        let w = r - l;

        if h <= zero || w <= zero {
            return None;
        }

        Some(TLBR::from_tlbr(t, l, b, r).unwrap())
    }

    fn intersection_area_with<R>(&self, other: &R) -> Self::Type
    where
        Self::Type: Float,
        R: Rect<Type = Self::Type, Unit = Self::Unit>,
    {
        self.intersect_with(other)
            .map(|rect| rect.area())
            .unwrap_or_else(Self::Type::zero)
    }

    fn iou_with<R>(&self, other: &R) -> Self::Type
    where
        Self::Type: Float,
        R: Rect<Type = Self::Type, Unit = Self::Unit>,
    {
        let inter_area = self.intersection_area_with(other);
        let union_area = self.area() + other.area() - inter_area
            + <Self::Type as NumCast>::from(EPSILON).unwrap();
        inter_area / union_area
    }

    fn hausdorff_distance_to<R>(&self, other: &R) -> Self::Type
    where
        Self::Type: Float,
        R: Rect<Type = Self::Type, Unit = Self::Unit>,
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
