use crate::{common::*, PixelCyCxHW, PixelTLBR, Rect};
use opencv::core as core_cv;

impl<T> TryFrom<&core_cv::Rect_<T>> for PixelTLBR<T>
where
    T: Num + Copy + PartialOrd,
{
    type Error = Error;

    fn try_from(from: &core_cv::Rect_<T>) -> Result<Self, Self::Error> {
        let core_cv::Rect_ {
            x: l,
            y: t,
            width: w,
            height: h,
        } = *from;
        Self::from_tlhw(t, l, h, w)
    }
}

impl<T> TryFrom<core_cv::Rect_<T>> for PixelTLBR<T>
where
    T: Num + Copy + PartialOrd,
{
    type Error = Error;

    fn try_from(from: core_cv::Rect_<T>) -> Result<Self, Self::Error> {
        (&from).try_into()
    }
}

impl<T> TryFrom<&core_cv::Rect_<T>> for PixelCyCxHW<T>
where
    T: Num + Copy + PartialOrd,
{
    type Error = Error;

    fn try_from(from: &core_cv::Rect_<T>) -> Result<Self, Self::Error> {
        let core_cv::Rect_ {
            x: l,
            y: t,
            width: w,
            height: h,
        } = *from;
        Self::from_tlhw(t, l, h, w)
    }
}

impl<T> TryFrom<core_cv::Rect_<T>> for PixelCyCxHW<T>
where
    T: Num + Copy + PartialOrd,
{
    type Error = Error;

    fn try_from(from: core_cv::Rect_<T>) -> Result<Self, Self::Error> {
        (&from).try_into()
    }
}

impl<T> From<&PixelTLBR<T>> for core_cv::Rect_<T>
where
    T: Num + Copy,
{
    fn from(from: &PixelTLBR<T>) -> Self {
        Self {
            x: from.l(),
            y: from.t(),
            width: from.w(),
            height: from.h(),
        }
    }
}

impl<T> From<PixelTLBR<T>> for core_cv::Rect_<T>
where
    T: Num + Copy,
{
    fn from(from: PixelTLBR<T>) -> Self {
        (&from).into()
    }
}

impl<T> From<&PixelCyCxHW<T>> for core_cv::Rect_<T>
where
    T: Num + Copy,
{
    fn from(from: &PixelCyCxHW<T>) -> Self {
        Self {
            x: from.l(),
            y: from.t(),
            width: from.w(),
            height: from.h(),
        }
    }
}

impl<T> From<PixelCyCxHW<T>> for core_cv::Rect_<T>
where
    T: Num + Copy,
{
    fn from(from: PixelCyCxHW<T>) -> Self {
        (&from).into()
    }
}
