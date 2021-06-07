use crate::{
    bbox::{CyCxHW, TLBR},
    common::*,
    size::Size,
    unit::{GridUnit, PixelUnit, RatioUnit, Unit},
};

/// Generic bounding box with an extra class ID.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Label<T, U>
where
    T: Float,
    U: Unit,
{
    pub cycxhw: CyCxHW<T, U>,
    pub class: usize,
}

pub type RatioLabel = Label<R64, RatioUnit>;
pub type PixelLabel = Label<R64, PixelUnit>;
pub type GridLabel = Label<R64, GridUnit>;

impl<T, U> Label<T, U>
where
    T: Float,
    U: Unit,
{
    pub fn cy(&self) -> T {
        self.cycxhw.cy()
    }

    pub fn cx(&self) -> T {
        self.cycxhw.cx()
    }

    pub fn h(&self) -> T {
        self.cycxhw.h()
    }

    pub fn w(&self) -> T {
        self.cycxhw.w()
    }

    pub fn tlbr(&self) -> TLBR<T, U> {
        (&self.cycxhw).into()
    }

    pub fn size(&self) -> Size<T, U> {
        self.cycxhw.size()
    }

    pub fn area(&self) -> T {
        self.cycxhw.area()
    }

    /// Compute intersection area in TLBR format.
    pub fn intersect_with(&self, other: &CyCxHW<T, U>) -> Option<Self> {
        let intersection: CyCxHW<_, _> =
            (&TLBR::from(&self.cycxhw).intersect_with(&other.into())?).into();

        Some(Self {
            cycxhw: intersection,
            class: self.class,
        })
    }

    pub fn scale_size(&self, scale: T) -> Result<Self> {
        let Self { ref cycxhw, class } = *self;
        Ok(Self {
            cycxhw: cycxhw.scale_size(scale)?,
            class,
        })
    }
}

impl<T, U> AsRef<CyCxHW<T, U>> for Label<T, U>
where
    T: Float,
    U: Unit,
{
    fn as_ref(&self) -> &CyCxHW<T, U> {
        &self.cycxhw
    }
}
