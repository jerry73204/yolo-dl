use bbox::{CyCxHW, Rect, Transform, TLBR};
use num_traits::Num;
use std::ops::Mul;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Label<R, C>
where
    R: Rect,
{
    pub rect: R,
    pub class: C,
}

impl<'a, T, C> Mul<&'a Label<TLBR<T>, C>> for &'a Transform<T>
where
    T: Copy + Num + PartialOrd,
    C: Copy,
{
    type Output = Label<TLBR<T>, C>;

    fn mul(self, rhs: &'a Label<TLBR<T>, C>) -> Self::Output {
        Label {
            rect: self * &rhs.rect,
            class: rhs.class,
        }
    }
}

impl<'a, T, C> Mul<&'a Label<CyCxHW<T>, C>> for &'a Transform<T>
where
    T: Copy + Num + PartialOrd,
    C: Copy,
{
    type Output = Label<CyCxHW<T>, C>;

    fn mul(self, rhs: &'a Label<CyCxHW<T>, C>) -> Self::Output {
        Label {
            rect: self * &rhs.rect,
            class: rhs.class,
        }
    }
}
