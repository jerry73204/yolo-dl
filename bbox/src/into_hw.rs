use crate::{common::*, HW};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HW_<T> {
    pub h: T,
    pub w: T,
}

impl<T> TryFrom<HW_<T>> for HW<T>
where
    T: Copy + Num + PartialOrd,
{
    type Error = anyhow::Error;

    fn try_from(from: HW_<T>) -> Result<Self, Self::Error> {
        (&from).try_into()
    }
}

impl<T> TryFrom<&HW_<T>> for HW<T>
where
    T: Copy + Num + PartialOrd,
{
    type Error = anyhow::Error;

    fn try_from(from: &HW_<T>) -> Result<Self, Self::Error> {
        let HW_ { h, w } = *from;
        HW::try_from_hw([h, w])
    }
}
