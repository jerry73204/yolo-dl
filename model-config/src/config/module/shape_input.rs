use super::*;
use crate::{common::*, config::misc::Shape};

#[derive(Debug, Clone, PartialEq, Eq, Derivative)]
#[derivative(Hash)]
pub enum ShapeInput<'a> {
    None,
    PlaceHolder,
    SingleDetect2D,
    SingleMergeDetect2D,
    IndexedDetect2D(usize),
    SingleTensor(&'a Shape),
    IndexedTensors(Vec<&'a Shape>),
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

    pub fn single_tensor(&self) -> Option<&Shape> {
        match self {
            Self::SingleTensor(shape) => Some(shape),
            _ => None,
        }
    }

    pub fn is_single_detect_2d(&self) -> bool {
        matches!(self, Self::SingleDetect2D)
    }

    pub fn indexed_tensors(&self) -> Option<&[&Shape]> {
        match self {
            Self::IndexedTensors(shapes) => Some(shapes),
            _ => None,
        }
    }

    pub fn is_indexed_detect_2d(&self) -> bool {
        matches!(self, Self::IndexedDetect2D(_))
    }
}

impl<'a> From<&'a Shape> for ShapeInput<'a> {
    fn from(from: &'a Shape) -> Self {
        Self::SingleTensor(from)
    }
}

impl<'a> TryFrom<&'a ShapeOutput> for ShapeInput<'a> {
    type Error = Error;

    fn try_from(from: &'a ShapeOutput) -> Result<Self, Self::Error> {
        let input_shape: Self = match from {
            ShapeOutput::Shape(shape) => shape.into(),
            ShapeOutput::Detect2D => Self::SingleDetect2D,
            ShapeOutput::MergeDetect2D => Self::SingleMergeDetect2D,
        };
        Ok(input_shape)
    }
}

impl<'a, 'b> From<&'b [&'a Shape]> for ShapeInput<'a> {
    fn from(from: &'b [&'a Shape]) -> Self {
        Self::IndexedTensors(from.into())
    }
}

impl<'a, 'b> TryFrom<&'b [&'a ShapeOutput]> for ShapeInput<'a> {
    type Error = Error;

    fn try_from(from: &'b [&'a ShapeOutput]) -> Result<Self, Self::Error> {
        ensure!(!from.is_empty(), "input shape list must not be empty");

        let shape = match from[0] {
            ShapeOutput::Shape(_) => {
                let shapes: Option<Vec<_>> = from.iter().map(|shape| shape.tensor()).collect();
                let shapes =
                    shapes.ok_or_else(|| format_err!("invalid input shape list {:?}", from))?;
                Self::IndexedTensors(shapes)
            }
            ShapeOutput::Detect2D => {
                ensure!(
                    from.iter().all(|shape| shape.is_detect_2d()),
                    "invalid input shape list {:?}",
                    from
                );
                Self::IndexedDetect2D(from.len())
            }
            ShapeOutput::MergeDetect2D => {
                bail!("invalid input shape {:?}", from)
            }
        };

        Ok(shape)
    }
}
