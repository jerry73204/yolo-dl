use crate::{common::*, dim::Dim};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Shape(Vec<Dim>);

impl Shape {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn size0(&self) -> Option<()> {
        match self.as_ref() {
            &[] => Some(()),
            _ => None,
        }
    }

    pub fn size1(&self) -> Option<Dim> {
        match self.as_ref() {
            &[size] => Some(size),
            _ => None,
        }
    }

    pub fn size2(&self) -> Option<[Dim; 2]> {
        match self.as_ref() {
            &[s1, s2] => Some([s1, s2]),
            _ => None,
        }
    }

    pub fn size3(&self) -> Option<[Dim; 3]> {
        match self.as_ref() {
            &[s1, s2, s3] => Some([s1, s2, s3]),
            _ => None,
        }
    }

    pub fn size4(&self) -> Option<[Dim; 4]> {
        match self.as_ref() {
            &[s1, s2, s3, s4] => Some([s1, s2, s3, s4]),
            _ => None,
        }
    }

    pub fn size5(&self) -> Option<[Dim; 5]> {
        match self.as_ref() {
            &[s1, s2, s3, s4, s5] => Some([s1, s2, s3, s4, s5]),
            _ => None,
        }
    }

    pub fn is_compatible_with(&self, other: &Shape) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }

        self.0
            .iter()
            .zip(other.0.iter())
            .all(|(lhs, rhs)| lhs.is_compatible_with(rhs))
    }

    pub fn equalize(&self, other: &Shape) -> Option<Shape> {
        if self.0.len() != other.0.len() {
            return None;
        }

        let new_shape: Option<Vec<Dim>> = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(lhs, rhs)| lhs.equalize(rhs))
            .collect();
        let new_shape = new_shape?;
        Some(new_shape.into())
    }
}

macro_rules! impl_tuple {
        ($(($ty:ident, $arg:ident)),*) => {
            impl<$($ty),*> From<( $($ty,)* )> for Shape
            where
                $(Dim: From<$ty>),*
            {
                fn from(( $($arg,)* ): ( $($ty,)* )) -> Self {
                    Shape(
                        vec![ $(Dim::from($arg)),* ]
                    )
                }
            }
        };
    }

impl_tuple!();
impl_tuple!((T0, t0));
impl_tuple!((T0, t0), (T1, t1));
impl_tuple!((T0, t0), (T1, t1), (T2, t2));
impl_tuple!((T0, t0), (T1, t1), (T2, t2), (T3, t3));
impl_tuple!((T0, t0), (T1, t1), (T2, t2), (T3, t3), (T4, t4));
impl_tuple!((T0, t0), (T1, t1), (T2, t2), (T3, t3), (T4, t4), (T5, t5));
impl_tuple!(
    (T0, t0),
    (T1, t1),
    (T2, t2),
    (T3, t3),
    (T4, t4),
    (T5, t5),
    (T6, t6)
);
impl_tuple!(
    (T0, t0),
    (T1, t1),
    (T2, t2),
    (T3, t3),
    (T4, t4),
    (T5, t5),
    (T6, t6),
    (T7, t7)
);
impl_tuple!(
    (T0, t0),
    (T1, t1),
    (T2, t2),
    (T3, t3),
    (T4, t4),
    (T5, t5),
    (T6, t6),
    (T7, t7),
    (T8, t8)
);
impl_tuple!(
    (T0, t0),
    (T1, t1),
    (T2, t2),
    (T3, t3),
    (T4, t4),
    (T5, t5),
    (T6, t6),
    (T7, t7),
    (T8, t8),
    (T9, t9)
);
impl_tuple!(
    (T0, t0),
    (T1, t1),
    (T2, t2),
    (T3, t3),
    (T4, t4),
    (T5, t5),
    (T6, t6),
    (T7, t7),
    (T8, t8),
    (T9, t9),
    (T10, t10)
);

impl<const SIZE: usize> From<[Dim; SIZE]> for Shape {
    fn from(from: [Dim; SIZE]) -> Self {
        Self(from.into())
    }
}

impl<const SIZE: usize> From<[usize; SIZE]> for Shape {
    fn from(from: [usize; SIZE]) -> Self {
        let slice: &[_] = from.as_ref();
        Self(slice.iter().cloned().map(Dim::from).collect())
    }
}

impl<const SIZE: usize> From<[Option<usize>; SIZE]> for Shape {
    fn from(from: [Option<usize>; SIZE]) -> Self {
        let slice: &[_] = from.as_ref();
        Self(slice.iter().cloned().map(Dim::from).collect())
    }
}

impl From<Vec<Option<usize>>> for Shape {
    fn from(vec: Vec<Option<usize>>) -> Self {
        Self(vec.into_iter().map(Dim::from).collect())
    }
}

impl From<&[Option<usize>]> for Shape {
    fn from(slice: &[Option<usize>]) -> Self {
        Vec::from(slice).into()
    }
}

impl From<Vec<usize>> for Shape {
    fn from(vec: Vec<usize>) -> Self {
        Self(vec.into_iter().map(Dim::from).collect())
    }
}

impl From<&Shape> for Shape {
    fn from(from: &Shape) -> Self {
        from.clone()
    }
}

impl From<Vec<Dim>> for Shape {
    fn from(vec: Vec<Dim>) -> Self {
        Self(vec)
    }
}

impl From<&[usize]> for Shape {
    fn from(slice: &[usize]) -> Self {
        Vec::from(slice).into()
    }
}

impl From<&Shape> for Vec<Option<usize>> {
    fn from(shape: &Shape) -> Self {
        shape.0.iter().cloned().map(Into::into).collect()
    }
}

impl From<&Shape> for Vec<Dim> {
    fn from(shape: &Shape) -> Self {
        shape.0.to_vec()
    }
}

impl<const SIZE: usize> TryFrom<&Shape> for [Option<usize>; SIZE] {
    type Error = &'static str;

    fn try_from(shape: &Shape) -> Result<Self, Self::Error> {
        if shape.len() != SIZE {
            return Err("shape mismatch");
        }
        let mut output = [None; SIZE];
        shape.0.iter().enumerate().for_each(|(index, &dim)| {
            output[index] = dim.into();
        });
        Ok(output)
    }
}

impl<const SIZE: usize> TryFrom<&Shape> for [Dim; SIZE] {
    type Error = &'static str;

    fn try_from(shape: &Shape) -> Result<Self, Self::Error> {
        Self::try_from(shape.0.clone()).map_err(|_| "shape mismatch")
    }
}

impl<const SIZE: usize> TryFrom<&Shape> for [usize; SIZE] {
    type Error = &'static str;

    fn try_from(shape: &Shape) -> Result<Self, Self::Error> {
        if shape.len() != SIZE {
            return Err("shape mismatch");
        }
        shape
            .0
            .iter()
            .enumerate()
            .try_fold([0; SIZE], |mut output, (index, &dim)| -> Option<_> {
                let dim = dim.size()?;
                output[index] = dim;
                Some(output)
            })
            .ok_or("shape cannot be fully determined")
    }
}

impl TryFrom<&Shape> for Vec<usize> {
    type Error = &'static str;

    fn try_from(shape: &Shape) -> Result<Self, Self::Error> {
        shape
            .0
            .iter()
            .try_fold(vec![], |mut output, &dim| -> Option<_> {
                output.push(dim.size()?);
                Some(output)
            })
            .ok_or("shape cannot be fully determined")
    }
}

impl FromIterator<usize> for Shape {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = usize>,
    {
        let shape: Vec<Dim> = iter.into_iter().map(Into::into).collect();
        Self(shape)
    }
}

impl AsRef<[Dim]> for Shape {
    fn as_ref(&self) -> &[Dim] {
        &self.0
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();
        self.0.iter().for_each(|dim| match dim.size() {
            Some(value) => {
                list.entry(&value);
            }
            None => {
                struct PlaceHolder;

                impl fmt::Debug for PlaceHolder {
                    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        f.write_str("_")
                    }
                }

                list.entry(&PlaceHolder);
            }
        });
        list.finish()
    }
}
