use crate::{element::Element, rect::Rect, CyCxHW};

/// Unchecked bounding box in CyCxHW format.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CyCxHW_<T>
where
    T: Element,
{
    pub cy: T,
    pub cx: T,
    pub h: T,
    pub w: T,
}

impl<T> TryFrom<&CyCxHW_<T>> for CyCxHW<T>
where
    T: Element,
{
    type Error = anyhow::Error;

    fn try_from(from: &CyCxHW_<T>) -> Result<Self, Self::Error> {
        let CyCxHW_ { cy, cx, h, w } = *from;
        Self::try_from_cycxhw([cy, cx, h, w])
    }
}

impl<T> TryFrom<CyCxHW_<T>> for CyCxHW<T>
where
    T: Element,
{
    type Error = anyhow::Error;

    fn try_from(from: CyCxHW_<T>) -> Result<Self, Self::Error> {
        Self::try_from(&from)
    }
}
