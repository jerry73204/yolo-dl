use crate::{
    common::*,
    unit::{GridUnit, PixelUnit, RatioUnit, Unit},
};
use num_traits::NumCast;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, TensorLike)]
pub struct HW<T> {
    pub h: T,
    pub w: T,
}

/// Generic size type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, TensorLike)]
pub struct Size<T, U>
where
    U: Unit,
{
    inner: HW<T>,
    #[tensor_like(copy)]
    _phantom: PhantomData<U>,
}

impl<T, U> Size<T, U>
where
    U: Unit,
{
    pub fn from_hw(h: T, w: T) -> Result<Self>
    where
        T: Num + PartialOrd,
    {
        HW { h, w }.try_into()
    }

    pub fn hw(&self) -> [T; 2]
    where
        T: Copy,
    {
        [self.h, self.w]
    }

    pub fn cast<S>(&self) -> Option<Size<S, U>>
    where
        T: Copy + ToPrimitive,
        S: Num + NumCast,
    {
        let h = <S as NumCast>::from(self.h)?;
        let w = <S as NumCast>::from(self.w)?;
        Some(Size {
            inner: HW { h, w },
            _phantom: PhantomData,
        })
    }

    pub fn area(&self) -> T
    where
        T: Num + Copy,
    {
        let Self {
            inner: HW { h, w }, ..
        } = *self;
        h * w
    }
}

impl<T, U> TryFrom<HW<T>> for Size<T, U>
where
    T: Num + PartialOrd,
    U: Unit,
{
    type Error = Error;

    fn try_from(from: HW<T>) -> Result<Self, Self::Error> {
        let zero = T::zero();
        ensure!(
            from.h >= zero && from.w >= zero,
            "the height and width must be non-negative"
        );

        Ok(Self {
            inner: from,
            _phantom: PhantomData,
        })
    }
}

impl<T, U> From<Size<T, U>> for HW<T>
where
    U: Unit,
{
    fn from(from: Size<T, U>) -> Self {
        from.inner
    }
}

impl<T, U> Deref for Size<T, U>
where
    U: Unit,
{
    type Target = HW<T>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub type PixelSize<T> = Size<T, PixelUnit>;
pub type GridSize<T> = Size<T, GridUnit>;
pub type RatioSize<T> = Size<T, RatioUnit>;
