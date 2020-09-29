use super::{GridUnit, PixelUnit, Unit};
use crate::common::*;

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
}

pub type PixelSize<T> = Size<T, PixelUnit>;
pub type GridSize<T> = Size<T, GridUnit>;
