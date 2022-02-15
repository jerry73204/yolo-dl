use crate::{element::Element, rect::Rect, TLBR};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TLBR_<T> {
    pub t: T,
    pub l: T,
    pub b: T,
    pub r: T,
}

impl<T> TryFrom<TLBR_<T>> for TLBR<T>
where
    T: Element,
{
    type Error = anyhow::Error;

    fn try_from(from: TLBR_<T>) -> Result<Self, Self::Error> {
        Self::try_from(&from)
    }
}

impl<T> TryFrom<&TLBR_<T>> for TLBR<T>
where
    T: Element,
{
    type Error = anyhow::Error;

    fn try_from(from: &TLBR_<T>) -> Result<Self, Self::Error> {
        let TLBR_ { t, l, b, r } = *from;
        Self::try_from_tlbr([t, l, b, r])
    }
}
