use crate::{
    common::*, size::PixelSize, unit::Unit, CyCxHW, GridCyCxHW, PixelCyCxHW, RatioCyCxHW, Rect,
};

/// Generic bounding box with an extra class ID.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RectLabel<R>
where
    R: Rect,
{
    pub rect: R,
    pub class: usize,
}

pub type RatioRectLabel<T> = RectLabel<RatioCyCxHW<T>>;
pub type PixelRectLabel<T> = RectLabel<PixelCyCxHW<T>>;
pub type GridRectLabel<T> = RectLabel<GridCyCxHW<T>>;

impl<R> Rect for RectLabel<R>
where
    R: Rect,
{
    type Type = R::Type;

    type Unit = R::Unit;

    fn t(&self) -> Self::Type {
        self.rect.t()
    }

    fn l(&self) -> Self::Type {
        self.rect.l()
    }

    fn b(&self) -> Self::Type {
        self.rect.b()
    }

    fn r(&self) -> Self::Type {
        self.rect.r()
    }

    fn cy(&self) -> Self::Type {
        self.rect.cy()
    }

    fn cx(&self) -> Self::Type {
        self.rect.cx()
    }

    fn h(&self) -> Self::Type {
        self.rect.h()
    }

    fn w(&self) -> Self::Type {
        self.rect.w()
    }
}

impl<T, U> RectLabel<CyCxHW<T, U>>
where
    T: Copy + Num,
    U: Unit,
{
    pub fn cast<V>(&self) -> Option<RectLabel<CyCxHW<V, U>>>
    where
        T: Copy + ToPrimitive,
        V: Copy + Num + NumCast,
    {
        Some(RectLabel {
            rect: self.rect.cast::<V>()?,
            class: self.class,
        })
    }
}

impl<T> PixelRectLabel<T>
where
    T: Copy + Num,
{
    pub fn to_ratio_label(&self, size: &PixelSize<T>) -> RatioRectLabel<T> {
        RatioRectLabel {
            rect: self.rect.to_ratio_cycxhw(size),
            class: self.class,
        }
    }
}

impl<T> RatioRectLabel<T>
where
    T: Copy + Num,
{
    pub fn to_pixel_label(&self, size: &PixelSize<T>) -> PixelRectLabel<T> {
        PixelRectLabel {
            rect: self.rect.to_pixel_cycxhw(size),
            class: self.class,
        }
    }
}
