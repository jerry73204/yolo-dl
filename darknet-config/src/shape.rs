use crate::common::*;

pub use shape_::*;
mod shape_ {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum Shape {
        Dim3([usize; 3]),
        Dim1(usize),
    }

    impl Shape {
        pub fn ndims(&self) -> usize {
            match self {
                Shape::Dim3(_) => 3,
                Shape::Dim1(_) => 1,
            }
        }

        pub fn iter(&self) -> impl Iterator<Item = usize> {
            Vec::<usize>::from(self).into_iter()
        }

        pub fn dim3(&self) -> Option<[usize; 3]> {
            match *self {
                Self::Dim3(hwc) => Some(hwc),
                Self::Dim1(_) => None,
            }
        }

        pub fn dim1(&self) -> Option<[usize; 1]> {
            match *self {
                Self::Dim1(flat) => Some([flat]),
                Self::Dim3(_) => None,
            }
        }
    }

    impl From<[usize; 1]> for Shape {
        fn from(from: [usize; 1]) -> Self {
            Self::Dim1(from[0])
        }
    }

    impl From<[usize; 3]> for Shape {
        fn from(from: [usize; 3]) -> Self {
            Self::Dim3(from)
        }
    }

    impl From<&Shape> for Vec<usize> {
        fn from(from: &Shape) -> Self {
            match *from {
                Shape::Dim3(hwc) => Vec::from(hwc),
                Shape::Dim1(flat) => vec![flat],
            }
        }
    }
    impl From<Shape> for Vec<usize> {
        fn from(from: Shape) -> Self {
            Self::from(&from)
        }
    }

    impl Display for Shape {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Dim3([h, w, c]) => f.debug_list().entries(vec![h, w, c]).finish(),
                Self::Dim1(size) => write!(f, "{}", size),
            }
        }
    }
}

pub use input_shape::*;
mod input_shape {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum InputShape {
        Single(Shape),
        Multiple(Vec<Shape>),
    }

    impl InputShape {
        pub fn single_dim1(&self) -> Option<[usize; 1]> {
            self.single()?.dim1()
        }

        pub fn single_dim3(&self) -> Option<[usize; 3]> {
            self.single()?.dim3()
        }

        pub fn single(&self) -> Option<Shape> {
            match *self {
                InputShape::Single(shape) => Some(shape),
                InputShape::Multiple(_) => None,
            }
        }

        pub fn multiple(&self) -> Option<&[Shape]> {
            match self {
                InputShape::Single(_) => None,
                InputShape::Multiple(shapes) => Some(&*shapes),
            }
        }

        pub fn multiple_dim3(&self) -> Option<Vec<[usize; 3]>> {
            let shapes = match self {
                InputShape::Single(_) => return None,
                InputShape::Multiple(shapes) => shapes,
            };

            shapes.iter().map(|shape| shape.dim3()).collect()
        }
    }

    impl From<Vec<Shape>> for InputShape {
        fn from(v: Vec<Shape>) -> Self {
            InputShape::Multiple(v)
        }
    }

    impl From<Shape> for InputShape {
        fn from(v: Shape) -> Self {
            InputShape::Single(v)
        }
    }

    impl From<[usize; 1]> for InputShape {
        fn from(v: [usize; 1]) -> Self {
            InputShape::Single(v.into())
        }
    }

    impl From<[usize; 3]> for InputShape {
        fn from(v: [usize; 3]) -> Self {
            InputShape::Single(v.into())
        }
    }

    impl From<Vec<[usize; 3]>> for InputShape {
        fn from(from: Vec<[usize; 3]>) -> Self {
            let shape: Vec<Shape> = from.into_iter().map(|shape| shape.into()).collect();
            InputShape::Multiple(shape)
        }
    }
}

pub use output_shape::*;
mod output_shape {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub enum OutputShape {
        Shape(Shape),
        Yolo([usize; 3]),
    }

    impl OutputShape {
        pub fn shape(&self) -> Option<Shape> {
            match *self {
                OutputShape::Shape(shape) => Some(shape),
                OutputShape::Yolo(_) => None,
            }
        }

        pub fn dim1(&self) -> Option<[usize; 1]> {
            self.shape()?.dim1()
        }

        pub fn dim3(&self) -> Option<[usize; 3]> {
            self.shape()?.dim3()
        }

        pub fn yolo_dim3(&self) -> Option<[usize; 3]> {
            match *self {
                OutputShape::Shape(_) => None,
                OutputShape::Yolo(shape) => Some(shape),
            }
        }
    }

    impl From<Shape> for OutputShape {
        fn from(v: Shape) -> Self {
            OutputShape::Shape(v)
        }
    }

    impl From<[usize; 1]> for OutputShape {
        fn from(from: [usize; 1]) -> Self {
            OutputShape::Shape(Shape::from(from))
        }
    }

    impl From<[usize; 3]> for OutputShape {
        fn from(from: [usize; 3]) -> Self {
            OutputShape::Shape(Shape::from(from))
        }
    }
}
