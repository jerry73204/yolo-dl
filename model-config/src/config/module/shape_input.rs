use super::*;
use crate::{common::*, config::misc::Shape};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Derivative)]
#[derivative(Hash)]
pub enum ShapeKind<'a> {
    Tensor(&'a Shape),
    Detect2D,
}

impl ShapeKind<'_> {
    pub fn tensor(&self) -> Option<&Shape> {
        match self {
            Self::Tensor(shape) => Some(shape),
            _ => None,
        }
    }

    pub fn is_detect_2d(&self) -> bool {
        match self {
            Self::Detect2D => true,
            _ => false,
        }
    }
}

impl<'a> From<&'a Shape> for ShapeKind<'a> {
    fn from(from: &'a Shape) -> Self {
        Self::Tensor(from)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Derivative)]
#[derivative(Hash)]
pub enum ShapeInput<'a> {
    None,
    PlaceHolder,
    Single(ShapeKind<'a>),
    Indexed(Vec<ShapeKind<'a>>),
}

impl ShapeInput<'_> {
    pub fn is_none(&self) -> bool {
        match self {
            Self::None => true,
            _ => false,
        }
    }

    pub fn is_placeholder(&self) -> bool {
        match self {
            Self::PlaceHolder => true,
            _ => false,
        }
    }

    pub fn tensor(&self) -> Option<&Shape> {
        match self {
            Self::Single(ShapeKind::Tensor(shape)) => Some(shape),
            _ => None,
        }
    }

    pub fn is_detect_2d(&self) -> bool {
        match self {
            Self::Single(ShapeKind::Detect2D) => true,
            _ => false,
        }
    }

    pub fn indexed_tensor(&self) -> Option<Vec<&Shape>> {
        match self {
            Self::Indexed(vec) => {
                let shapes: Option<Vec<_>> = vec.iter().map(|kind| kind.tensor()).collect();
                shapes
            }
            _ => None,
        }
    }

    pub fn is_indexed_detect_2d(&self) -> bool {
        match self {
            Self::Indexed(vec) => vec.iter().all(|kind| kind.is_detect_2d()),
            _ => false,
        }
    }
}

impl<'a> From<&'a Shape> for ShapeInput<'a> {
    fn from(from: &'a Shape) -> Self {
        Self::Single(from.into())
    }
}

impl<'a> TryFrom<&'a ShapeOutput> for ShapeInput<'a> {
    type Error = Error;

    fn try_from(from: &'a ShapeOutput) -> Result<Self, Self::Error> {
        let input_shape: Self = match from {
            ShapeOutput::Shape(shape) => shape.into(),
            ShapeOutput::Detect2D => Self::Single(ShapeKind::Detect2D),
            ShapeOutput::MergeDetect2D => bail!("TODO"),
        };
        Ok(input_shape)
    }
}

impl<'a, 'b> From<&'b [&'a Shape]> for ShapeInput<'a> {
    fn from(from: &'b [&'a Shape]) -> Self {
        let shapes: Vec<ShapeKind> = from.iter().cloned().map(|shape| shape.into()).collect();
        Self::Indexed(shapes)
    }
}

impl<'a, 'b> TryFrom<&'b [&'a ShapeOutput]> for ShapeInput<'a> {
    type Error = Error;

    fn try_from(from: &'b [&'a ShapeOutput]) -> Result<Self, Self::Error> {
        let kinds: Vec<_> = from
            .iter()
            .map(|output_shape| -> Result<_> {
                let kind = match output_shape {
                    ShapeOutput::Shape(shape) => ShapeKind::Tensor(shape),
                    ShapeOutput::Detect2D => ShapeKind::Detect2D,
                    _ => bail!("TODO"),
                };
                Ok(kind)
            })
            .try_collect()?;
        Ok(Self::Indexed(kinds))
    }
}
