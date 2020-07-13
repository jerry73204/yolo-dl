use crate::common::*;

pub trait Unit {}

#[derive(Debug, Clone, Copy)]
pub struct Pixel;

impl Unit for Pixel {}

#[derive(Debug, Clone, Copy)]
pub struct Ratio;

impl Unit for Ratio {}

#[derive(Debug, Clone, Copy)]
pub struct Grid;

impl Unit for Grid {}

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Quantity<V, U>
where
    V: Copy
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Rem<Output = V>,
    U: Unit,
{
    value: V,
    _phantom: PhantomData<U>,
}

impl<V, U> Quantity<V, U>
where
    V: Copy
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Rem<Output = V>,
    U: Unit,
{
    pub fn new(value: V) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }

    pub fn get(&self) -> V {
        self.value
    }

    pub fn cast<NewV>(&self) -> Quantity<NewV, U>
    where
        NewV: Copy
            + Add<Output = NewV>
            + Sub<Output = NewV>
            + Mul<Output = NewV>
            + Div<Output = NewV>
            + Rem<Output = NewV>
            + From<V>,
    {
        Quantity {
            value: NewV::from(self.value),
            _phantom: PhantomData,
        }
    }

    pub fn to<NewU>(&self) -> Quantity<V, NewU>
    where
        NewU: Unit,
    {
        Quantity {
            value: self.value,
            _phantom: PhantomData,
        }
    }

    pub fn mul_to<NewU>(&self, scale: V) -> Quantity<V, NewU>
    where
        NewU: Unit,
    {
        Quantity {
            value: self.value * scale,
            _phantom: PhantomData,
        }
    }

    pub fn div_to<NewU>(&self, scale: V) -> Quantity<V, NewU>
    where
        NewU: Unit,
    {
        Quantity {
            value: self.value / scale,
            _phantom: PhantomData,
        }
    }
}

// deref

impl<V, U> Deref for Quantity<V, U>
where
    V: Copy
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Rem<Output = V>,
    U: Unit,
{
    type Target = V;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<V, U> DerefMut for Quantity<V, U>
where
    V: Copy
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Rem<Output = V>,
    U: Unit,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

// conversion

impl<V, U> From<V> for Quantity<V, U>
where
    V: Copy
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Rem<Output = V>,
    U: Unit,
{
    fn from(value: V) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

// add

impl<V, U> Add<Quantity<V, U>> for Quantity<V, U>
where
    V: Copy
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Rem<Output = V>,
    U: Unit,
{
    type Output = Quantity<V, U>;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Output {
            value: self.value + rhs.value,
            _phantom: PhantomData,
        }
    }
}

impl<V, U> Add<V> for Quantity<V, U>
where
    V: Copy
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Rem<Output = V>,
    U: Unit,
{
    type Output = Quantity<V, U>;

    fn add(self, rhs: V) -> Self::Output {
        Self::Output {
            value: self.value + rhs,
            _phantom: PhantomData,
        }
    }
}

// sub

impl<V, U> Sub<Quantity<V, U>> for Quantity<V, U>
where
    V: Copy
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Rem<Output = V>,
    U: Unit,
{
    type Output = Quantity<V, U>;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output {
            value: self.value - rhs.value,
            _phantom: PhantomData,
        }
    }
}

impl<V, U> Sub<V> for Quantity<V, U>
where
    V: Copy
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Rem<Output = V>,
    U: Unit,
{
    type Output = Quantity<V, U>;

    fn sub(self, rhs: V) -> Self::Output {
        Self::Output {
            value: self.value - rhs,
            _phantom: PhantomData,
        }
    }
}

// mul

impl<V, U> Mul<Quantity<V, U>> for Quantity<V, U>
where
    V: Copy
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Rem<Output = V>,
    U: Unit,
{
    type Output = Quantity<V, U>;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::Output {
            value: self.value * rhs.value,
            _phantom: PhantomData,
        }
    }
}

impl<V, U> Mul<V> for Quantity<V, U>
where
    V: Copy
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Rem<Output = V>,
    U: Unit,
{
    type Output = Quantity<V, U>;

    fn mul(self, rhs: V) -> Self::Output {
        Self::Output {
            value: self.value * rhs,
            _phantom: PhantomData,
        }
    }
}

// div

impl<V, U> Div<Quantity<V, U>> for Quantity<V, U>
where
    V: Copy
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Rem<Output = V>,
    U: Unit,
{
    type Output = Quantity<V, U>;

    fn div(self, rhs: Self) -> Self::Output {
        Self::Output {
            value: self.value / rhs.value,
            _phantom: PhantomData,
        }
    }
}

impl<V, U> Div<V> for Quantity<V, U>
where
    V: Copy
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Rem<Output = V>,
    U: Unit,
{
    type Output = Quantity<V, U>;

    fn div(self, rhs: V) -> Self::Output {
        Self::Output {
            value: self.value / rhs,
            _phantom: PhantomData,
        }
    }
}

// rem

impl<V, U> Rem<Quantity<V, U>> for Quantity<V, U>
where
    V: Copy
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Rem<Output = V>,
    U: Unit,
{
    type Output = Quantity<V, U>;

    fn rem(self, rhs: Self) -> Self::Output {
        Self::Output {
            value: self.value % rhs.value,
            _phantom: PhantomData,
        }
    }
}

impl<V, U> Rem<V> for Quantity<V, U>
where
    V: Copy
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Div<Output = V>
        + Rem<Output = V>,
    U: Unit,
{
    type Output = Quantity<V, U>;

    fn rem(self, rhs: V) -> Self::Output {
        Self::Output {
            value: self.value % rhs,
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_type_test() {
        let pixels: Quantity<u32, Pixel> = 640.into();
        assert_eq!(*pixels, 640);

        let ratio: Quantity<f64, Ratio> = pixels.cast::<f64>().div_to(1000.0);
        assert_eq!(*ratio, 640 as f64 / 1000.0);
    }
}
