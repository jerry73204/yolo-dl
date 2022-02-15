use super::{CyCxHW, Rect, TLBR};
use crate::{common::*, element::Element, RectExt, HW};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Transform<T>
where
    T: Element,
{
    pub sy: T,
    pub sx: T,
    pub ty: T,
    pub tx: T,
}

impl<T> Transform<T>
where
    T: Element,
{
    pub fn from_rects<R>(src: &R, tgt: &R) -> Self
    where
        R: Rect<Type = T>,
    {
        let sy = tgt.h() / src.h();
        let sx = tgt.w() / src.w();
        let ty = tgt.t() - src.t() * sy;
        let tx = tgt.l() - src.l() * sx;

        Self { sy, sx, ty, tx }
    }

    pub fn from_sizes_exact<S>(src_size: S, tgt_size: S) -> Self
    where
        S: TryInto<HW<T>>,
    {
        let src_size = src_size.try_into().unwrap_or_else(|_| panic!());
        let tgt_size = tgt_size.try_into().unwrap_or_else(|_| panic!());
        let src = TLBR::from_tlhw([T::zero(), T::zero(), src_size.h(), src_size.w()]);
        let tgt = TLBR::from_tlhw([T::zero(), T::zero(), tgt_size.h(), tgt_size.w()]);
        Self::from_rects(&src, &tgt)
    }

    pub fn from_sizes_letterbox<S>(src_size: S, tgt_size: S) -> Self
    where
        S: TryInto<HW<T>>,
    {
        let src_size = src_size.try_into().unwrap_or_else(|_| panic!());
        let tgt_size = tgt_size.try_into().unwrap_or_else(|_| panic!());

        let (new_h, new_w) = if tgt_size.h() * src_size.w() <= tgt_size.w() * src_size.h() {
            let new_h = tgt_size.h();
            let new_w = src_size.w() * tgt_size.h() / src_size.h();
            (new_h, new_w)
        } else {
            let new_h = src_size.h() * tgt_size.w() / src_size.w();
            let new_w = tgt_size.w();
            (new_h, new_w)
        };

        let two = T::one() + T::one();
        let off_y = (tgt_size.h() - new_h) / two;
        let off_x = (tgt_size.w() - new_w) / two;

        let src = TLBR::from_tlhw([T::zero(), T::zero(), src_size.h(), src_size.w()]);
        let tgt = TLBR::from_tlhw([off_y, off_x, new_h, new_w]);

        Self::from_rects(&src, &tgt)
    }

    pub fn inverse(&self) -> Self {
        let sy = T::one() / self.sy;
        let sx = T::one() / self.sx;
        let ty = -self.ty / self.sy;
        let tx = -self.tx / self.sx;

        Self { sy, sx, ty, tx }
    }

    pub fn cast<V>(&self) -> Option<Transform<V>>
    where
        V: Element,
    {
        Some(Transform {
            sy: V::from(self.sy)?,
            sx: V::from(self.sx)?,
            ty: V::from(self.ty)?,
            tx: V::from(self.tx)?,
        })
    }
}

impl<T> Mul<&TLBR<T>> for &Transform<T>
where
    T: Element,
{
    type Output = TLBR<T>;

    fn mul(self, rhs: &TLBR<T>) -> Self::Output {
        TLBR {
            t: rhs.t * self.sy + self.ty,
            l: rhs.l * self.sx + self.tx,
            b: rhs.b * self.sy + self.ty,
            r: rhs.r * self.sx + self.tx,
        }
    }
}

impl<T> Mul<&CyCxHW<T>> for &Transform<T>
where
    T: Element,
{
    type Output = CyCxHW<T>;

    fn mul(self, rhs: &CyCxHW<T>) -> Self::Output {
        CyCxHW {
            cy: rhs.cy * self.sy + self.ty,
            cx: rhs.cx * self.sx + self.tx,
            h: rhs.h * self.sy,
            w: rhs.w * self.sx,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::HW_;

    use super::*;

    #[test]
    fn rect_transform_inverse() {
        let orig = Transform {
            sx: 2.0,
            sy: 2.0,
            tx: 1.0,
            ty: 1.0,
        };
        assert_eq!(orig.inverse().inverse(), orig);
    }

    #[test]
    fn rect_resize_exact() {
        let transform =
            Transform::from_sizes_exact(HW_ { h: 80.0, w: 80.0 }, HW_ { h: 20.0, w: 40.0 });
        let expect = Transform {
            sx: 0.5,
            sy: 0.25,
            tx: 0.0,
            ty: 0.0,
        };
        assert_eq!(transform, expect);
    }

    #[test]
    fn rect_resize_letterbox() {
        let transform =
            Transform::from_sizes_letterbox(HW_ { h: 80.0, w: 80.0 }, HW_ { h: 20.0, w: 40.0 });
        let expect = Transform {
            sx: 0.25,
            sy: 0.25,
            tx: 10.0,
            ty: 0.0,
        };
        assert_eq!(transform, expect);
    }
}
