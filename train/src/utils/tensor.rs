use crate::common::*;

// CowTensorSlice

/// A helper type to save either a borrowed or an owned [Tensor](tch::Tensor) type.
pub enum CowTensor<'a> {
    Borrowed(&'a Tensor),
    Owned(Tensor),
}

impl<'a> CowTensor<'a> {
    pub fn into_owned(self) -> Tensor {
        match self {
            Self::Borrowed(borrowed) => borrowed.shallow_clone(),
            Self::Owned(owned) => owned,
        }
    }
}

impl<'a> From<&'a Tensor> for CowTensor<'a> {
    fn from(from: &'a Tensor) -> Self {
        Self::Borrowed(from)
    }
}

impl<'a> From<Tensor> for CowTensor<'a> {
    fn from(from: Tensor) -> Self {
        Self::Owned(from)
    }
}
