use crate::{
    common::*,
    ratio::Ratio,
    unit::{GridUnit, PixelUnit, RatioUnit, Unit},
};

/// Generic size type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, TensorLike)]
pub struct Size<T, U>
where
    U: Unit,
{
    pub h: T,
    pub w: T,
    #[tensor_like(copy)]
    _phantom: PhantomData<U>,
}

impl<T, U> Size<T, U>
where
    U: Unit,
{
    pub fn new(h: T, w: T) -> Self {
        Self {
            h,
            w,
            _phantom: PhantomData,
        }
    }

    pub fn map<F, R>(&self, mut f: F) -> Size<R, U>
    where
        F: FnMut(&T) -> R,
    {
        Size {
            h: f(&self.h),
            w: f(&self.w),
            _phantom: PhantomData,
        }
    }
}

pub type PixelSize<T> = Size<T, PixelUnit>;
pub type GridSize<T> = Size<T, GridUnit>;
pub type RatioSize = Size<Ratio, RatioUnit>;
