use super::{CyCxHW, GridRectLabel, PixelRectLabel, RatioRectLabel, Rect, RectLabel, TLBR};
use crate::{
    common::*,
    size::{GridSize, PixelSize, Size},
    unit::{GridUnit, PixelUnit, RatioUnit, Unit},
};

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

impl<T> PixelRectTransform<T> {
    pub fn to_ratio_transform(&self, size: &PixelSize<T>) -> RatioRectTransform<T>
    where
        T: Num + Copy,
    {
        RatioRectTransform {
            sy: self.sy,
            sx: self.sx,
            ty: self.ty / size.h,
            tx: self.tx / size.w,
            _phantom: PhantomData,
        }
    }
}

impl<T> GridRectTransform<T> {
    pub fn to_ratio_transform(&self, size: &GridSize<T>) -> RatioRectTransform<T>
    where
        T: Num + Copy,
    {
        RatioRectTransform {
            sy: self.sy,
            sx: self.sx,
            ty: self.ty / size.h,
            tx: self.tx / size.w,
            _phantom: PhantomData,
        }
    }
}

impl<T> RatioRectTransform<T> {
    pub fn to_pixel_transform(&self, size: &PixelSize<T>) -> PixelRectTransform<T>
    where
        T: Num + Copy,
    {
        PixelRectTransform {
            sy: self.sy,
            sx: self.sx,
            ty: self.ty * size.h,
            tx: self.tx * size.w,
            _phantom: PhantomData,
        }
    }

    pub fn to_grid(&self, size: &GridSize<T>) -> GridRectTransform<T>
    where
        T: Num + Copy,
    {
        GridRectTransform {
            sy: self.sy,
            sx: self.sx,
            ty: self.ty * size.h,
            tx: self.tx * size.w,
            _phantom: PhantomData,
        }
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

impl<T> Mul<&PixelRectLabel<T>> for &PixelRectTransform<T>
where
    T: Num + Copy,
{
    type Output = PixelRectLabel<T>;

    fn mul(self, rhs: &PixelRectLabel<T>) -> Self::Output {
        RectLabel {
            rect: self * &rhs.rect,
            class: rhs.class,
        }
    }
}

impl<T> Mul<&GridRectLabel<T>> for &GridRectTransform<T>
where
    T: Num + Copy,
{
    type Output = GridRectLabel<T>;

    fn mul(self, rhs: &GridRectLabel<T>) -> Self::Output {
        RectLabel {
            rect: self * &rhs.rect,
            class: rhs.class,
        }
    }
}

impl<T> Mul<&RatioRectLabel<T>> for &RatioRectTransform<T>
where
    T: Num + Copy,
{
    type Output = RatioRectLabel<T>;

    fn mul(self, rhs: &RatioRectLabel<T>) -> Self::Output {
        RectLabel {
            rect: self * &rhs.rect,
            class: rhs.class,
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
