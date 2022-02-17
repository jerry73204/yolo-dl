use crate::{common::*, CyCxHW, Rect, TLBR};
use opencv::core as core_cv;

impl<T> TryFrom<&core_cv::Rect_<T>> for TLBR<T>
where
    T: Copy + PartialOrd + Num,
{
    type Error = anyhow::Error;

    fn try_from(from: &core_cv::Rect_<T>) -> Result<Self, Self::Error> {
        let core_cv::Rect_ {
            x: l,
            y: t,
            width: w,
            height: h,
        } = *from;
        Self::try_from_tlhw([t, l, h, w])
    }
}

impl<T> TryFrom<core_cv::Rect_<T>> for TLBR<T>
where
    T: Copy + PartialOrd + Num,
{
    type Error = anyhow::Error;

    fn try_from(from: core_cv::Rect_<T>) -> Result<Self, Self::Error> {
        (&from).try_into()
    }
}

impl<T> TryFrom<&core_cv::Rect_<T>> for CyCxHW<T>
where
    T: Copy + PartialOrd + Num,
{
    type Error = anyhow::Error;

    fn try_from(from: &core_cv::Rect_<T>) -> Result<Self, Self::Error> {
        let core_cv::Rect_ {
            x: l,
            y: t,
            width: w,
            height: h,
        } = *from;
        Self::try_from_tlhw([t, l, h, w])
    }
}

impl<T> TryFrom<core_cv::Rect_<T>> for CyCxHW<T>
where
    T: Copy + PartialOrd + Num,
{
    type Error = anyhow::Error;

    fn try_from(from: core_cv::Rect_<T>) -> Result<Self, Self::Error> {
        (&from).try_into()
    }
}

impl<T> From<&TLBR<T>> for core_cv::Rect_<T>
where
    T: Copy + PartialOrd + Num,
{
    fn from(from: &TLBR<T>) -> Self {
        Self {
            x: from.l(),
            y: from.t(),
            width: from.w(),
            height: from.h(),
        }
    }
}

impl<T> From<TLBR<T>> for core_cv::Rect_<T>
where
    T: Copy + PartialOrd + Num,
{
    fn from(from: TLBR<T>) -> Self {
        (&from).into()
    }
}

impl<T> From<&CyCxHW<T>> for core_cv::Rect_<T>
where
    T: Copy + PartialOrd + Num,
{
    fn from(from: &CyCxHW<T>) -> Self {
        Self {
            x: from.l(),
            y: from.t(),
            width: from.w(),
            height: from.h(),
        }
    }
}

impl<T> From<CyCxHW<T>> for core_cv::Rect_<T>
where
    T: Copy + PartialOrd + Num,
{
    fn from(from: CyCxHW<T>) -> Self {
        (&from).into()
    }
}
