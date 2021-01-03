use crate::{
    common::*,
    ratio::Ratio,
    unit::{GridUnit, PixelUnit, RatioUnit, Unit},
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Hash, Deserialize, TensorLike)]
pub struct Size<T, U>
where
    U: Unit,
{
    pub height: T,
    pub width: T,
    #[tensor_like(copy)]
    _phantom: PhantomData<U>,
}

impl<T, U> Size<T, U>
where
    U: Unit,
{
    pub fn new(height: T, width: T) -> Self {
        Self {
            height,
            width,
            _phantom: PhantomData,
        }
    }

    pub fn map<F, R>(&self, mut f: F) -> Size<R, U>
    where
        F: FnMut(&T) -> R,
    {
        Size {
            height: f(&self.height),
            width: f(&self.width),
            _phantom: PhantomData,
        }
    }
}

pub type PixelSize<T> = Size<T, PixelUnit>;
pub type GridSize<T> = Size<T, GridUnit>;
pub type RatioSize = Size<Ratio, RatioUnit>;
