use crate::{
    common::*,
    unit::{GridUnit, PixelUnit, RatioUnit, Unit},
};

/// Generic size type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, TensorLike, CopyGetters)]
pub struct Size<T, U>
where
    T: Zero + PartialOrd + Copy + ToPrimitive,
    U: Unit,
{
    #[get_copy = "pub"]
    h: T,
    #[get_copy = "pub"]
    w: T,
    #[tensor_like(copy)]
    _phantom: PhantomData<U>,
}

impl<T, U> Size<T, U>
where
    T: Zero + PartialOrd + Copy + ToPrimitive,
    U: Unit,
{
    pub fn new(h: T, w: T) -> Result<Self> {
        let zero = T::zero();
        ensure!(
            h >= zero && w >= zero,
            "the height and width must be non-negative"
        );

        Ok(Self {
            h,
            w,
            _phantom: PhantomData,
        })
    }

    pub fn hw_params(&self) -> [T; 2] {
        [self.h, self.w]
    }

    pub fn cast<S>(&self) -> Option<Size<S, U>>
    where
        S: NumCast + Zero + PartialOrd + Copy + ToPrimitive,
    {
        let h = <S as NumCast>::from(self.h)?;
        let w = <S as NumCast>::from(self.w)?;
        Some(Size {
            h,
            w,
            _phantom: PhantomData,
        })
    }
}

pub type PixelSize<T> = Size<T, PixelUnit>;
pub type GridSize<T> = Size<T, GridUnit>;
pub type RatioSize<T> = Size<T, RatioUnit>;
