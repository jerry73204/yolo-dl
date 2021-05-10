use crate::{
    common::*,
    config::misc::{Dim, Shape},
};

#[derive(Debug, Clone, PartialEq, Eq, Derivative, Serialize, Deserialize)]
#[derivative(Hash)]
pub enum ShapeOutput {
    Shape(Shape),
    Detect2D,
    MergeDetect2D,
}

impl ShapeOutput {
    pub fn tensor_nd<'a, Out>(&'a self) -> Result<Out>
    where
        Out: TryFrom<&'a Shape>,
    {
        match self {
            Self::Shape(shape) => shape
                .try_into()
                .map_err(|_| format_err!("shape mismatch or not fully determined")),
            _ => bail!("not a tensor"),
        }
    }

    pub fn tensor(&self) -> Option<&Shape> {
        match self {
            Self::Shape(shape) => Some(shape),
            _ => None,
        }
    }

    pub fn is_detect_2d(&self) -> bool {
        match self {
            Self::Detect2D => true,
            _ => false,
        }
    }

    pub fn is_merge_detect_2d(&self) -> bool {
        match self {
            Self::MergeDetect2D => true,
            _ => false,
        }
    }
}

impl From<Shape> for ShapeOutput {
    fn from(from: Shape) -> Self {
        Self::Shape(from)
    }
}

impl From<Vec<Dim>> for ShapeOutput {
    fn from(from: Vec<Dim>) -> Self {
        Self::Shape(Shape::from(from))
    }
}

impl From<Vec<usize>> for ShapeOutput {
    fn from(from: Vec<usize>) -> Self {
        Self::Shape(Shape::from(from))
    }
}

impl<const SIZE: usize> From<[usize; SIZE]> for ShapeOutput {
    fn from(from: [usize; SIZE]) -> Self {
        Self::Shape(Shape::from(from))
    }
}

impl Display for ShapeOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(shape) => Display::fmt(shape, f),
            Self::Detect2D => write!(f, "Detect2D"),
            Self::MergeDetect2D => write!(f, "MergeDetect2D"),
        }
    }
}
