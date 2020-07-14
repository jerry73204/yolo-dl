use crate::common::*;

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    derive_more::Mul,
    derive_more::Rem,
    derive_more::AddAssign,
    derive_more::SubAssign,
    derive_more::MulAssign,
    derive_more::RemAssign,
)]
#[repr(transparent)]
pub struct Ratio(R64);

impl Ratio {
    pub fn f_from_f64(value: f64) -> Result<Self> {
        Self::f_from_r64(R64::try_new(value).ok_or_else(|| format_err!("not a finite number"))?)
    }

    pub fn f_from_r64(value: R64) -> Result<Self> {
        ensure!(
            value.raw() >= 0.0 && value.raw() <= 1.0,
            "ratio value must be within range [0.0, 1.0]"
        );
        Ok(Self(value))
    }
}

impl Serialize for Ratio {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Ratio {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = f64::deserialize(deserializer)?;
        Self::f_from_f64(value).map_err(|err| D::Error::custom(format!("{:?}", err)))
    }
}

impl From<R64> for Ratio {
    fn from(value: R64) -> Self {
        Self::f_from_r64(value).unwrap()
    }
}

impl From<f64> for Ratio {
    fn from(value: f64) -> Self {
        Self::f_from_f64(value).unwrap()
    }
}

impl From<f32> for Ratio {
    fn from(value: f32) -> Self {
        Self::f_from_f64(value as f64).unwrap()
    }
}

impl From<Ratio> for R64 {
    fn from(ratio: Ratio) -> Self {
        ratio.0
    }
}

impl From<Ratio> for f64 {
    fn from(ratio: Ratio) -> Self {
        ratio.0.raw()
    }
}

impl From<Ratio> for f32 {
    fn from(ratio: Ratio) -> Self {
        ratio.0.raw() as f32
    }
}

impl Deref for Ratio {
    type Target = R64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Add for Ratio {
    type Output = Ratio;

    fn add(self, rhs: Self) -> Self::Output {
        let value = self.0 + rhs.0;
        Self::f_from_r64(value).unwrap()
    }
}

impl Sub for Ratio {
    type Output = Ratio;

    fn sub(self, rhs: Self) -> Self::Output {
        let value = self.0 - rhs.0;
        Self::f_from_r64(value).unwrap()
    }
}

// unit trait

// pub trait Unit {}

// #[derive(Debug, Clone, Copy)]
// pub struct PixelUnit;

// impl Unit for PixelUnit {}

// #[derive(Debug, Clone, Copy)]
// pub struct RatioUnit;

// impl Unit for RatioUnit {}

// #[derive(Debug, Clone, Copy)]
// pub struct GridUnit;

// impl Unit for GridUnit {}

// #[derive(Debug, Clone, Copy)]
// #[repr(transparent)]
// pub struct Quantity<V, U> {
//     value: V,
//     _phantom: PhantomData<U>,
// }

// impl<V, U> Quantity<V, U>
// where
//     U: Unit,
// {
//     pub fn new(value: V) -> Self {
//         Self {
//             value,
//             _phantom: PhantomData,
//         }
//     }

//     pub fn get(&self) -> V
//     where
//         V: Copy,
//     {
//         self.value
//     }

//     pub fn cast<NewV>(&self) -> Quantity<NewV, U>
//     where
//         V: Copy,
//         NewV: From<V>,
//     {
//         Quantity {
//             value: NewV::from(self.value),
//             _phantom: PhantomData,
//         }
//     }

//     pub fn to<NewU>(&self) -> Quantity<V, NewU>
//     where
//         V: Copy,
//         NewU: Unit,
//     {
//         Quantity {
//             value: self.value,
//             _phantom: PhantomData,
//         }
//     }

//     pub fn mul_to<NewU>(&self, scale: V) -> Quantity<V, NewU>
//     where
//         V: Copy + Mul<Output = V>,
//         NewU: Unit,
//     {
//         Quantity {
//             value: self.value * scale,
//             _phantom: PhantomData,
//         }
//     }

//     pub fn div_to<NewU>(&self, scale: V) -> Quantity<V, NewU>
//     where
//         V: Copy + Div<Output = V>,
//         NewU: Unit,
//     {
//         Quantity {
//             value: self.value / scale,
//             _phantom: PhantomData,
//         }
//     }
// }

// // deref

// impl<V, U> Deref for Quantity<V, U>
// where
//     U: Unit,
// {
//     type Target = V;

//     fn deref(&self) -> &Self::Target {
//         &self.value
//     }
// }

// impl<V, U> DerefMut for Quantity<V, U>
// where
//     U: Unit,
// {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.value
//     }
// }

// // conversion

// impl<V, U> From<V> for Quantity<V, U>
// where
//     U: Unit,
// {
//     fn from(value: V) -> Self {
//         Self {
//             value,
//             _phantom: PhantomData,
//         }
//     }
// }

// // hash

// impl<V, U> Hash for Quantity<V, U>
// where
//     V: Hash,
//     U: Unit,
// {
//     fn hash<H>(&self, state: &mut H)
//     where
//         H: Hasher,
//     {
//         self.hash(state)
//     }
// }

// // partial ord

// impl<V, U> PartialOrd<Quantity<V, U>> for Quantity<V, U>
// where
//     V: PartialOrd,
//     U: Unit,
// {
//     fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
//         self.value.partial_cmp(&rhs.value)
//     }
// }

// impl<V, U> PartialOrd<V> for Quantity<V, U>
// where
//     V: PartialOrd,
//     U: Unit,
// {
//     fn partial_cmp(&self, rhs: &V) -> Option<Ordering> {
//         self.value.partial_cmp(&rhs)
//     }
// }

// // ord

// impl<V, U> Ord for Quantity<V, U>
// where
//     V: Ord,
//     U: Unit,
// {
//     fn cmp(&self, rhs: &Self) -> Ordering {
//         self.value.cmp(&rhs.value)
//     }
// }

// // partial eq

// impl<V, U> PartialEq<Quantity<V, U>> for Quantity<V, U>
// where
//     V: PartialEq,
//     U: Unit,
// {
//     fn eq(&self, rhs: &Self) -> bool {
//         self.value.eq(&rhs.value)
//     }
// }

// impl<V, U> PartialEq<V> for Quantity<V, U>
// where
//     V: PartialEq,
//     U: Unit,
// {
//     fn eq(&self, rhs: &V) -> bool {
//         self.value.eq(&rhs)
//     }
// }

// // eq

// impl<V, U> Eq for Quantity<V, U>
// where
//     V: PartialEq,
//     U: Unit,
// {
// }

// // add

// impl<V, U> Add<Quantity<V, U>> for Quantity<V, U>
// where
//     V: Add<Output = V>,
//     U: Unit,
// {
//     type Output = Quantity<V, U>;

//     fn add(self, rhs: Self) -> Self::Output {
//         Self::Output {
//             value: self.value + rhs.value,
//             _phantom: PhantomData,
//         }
//     }
// }

// impl<V, U> Add<V> for Quantity<V, U>
// where
//     V: Add<Output = V>,
//     U: Unit,
// {
//     type Output = Quantity<V, U>;

//     fn add(self, rhs: V) -> Self::Output {
//         Self::Output {
//             value: self.value + rhs,
//             _phantom: PhantomData,
//         }
//     }
// }

// // sub

// impl<V, U> Sub<Quantity<V, U>> for Quantity<V, U>
// where
//     V: Sub<Output = V>,
//     U: Unit,
// {
//     type Output = Quantity<V, U>;

//     fn sub(self, rhs: Self) -> Self::Output {
//         Self::Output {
//             value: self.value - rhs.value,
//             _phantom: PhantomData,
//         }
//     }
// }

// impl<V, U> Sub<V> for Quantity<V, U>
// where
//     V: Sub<Output = V>,
//     U: Unit,
// {
//     type Output = Quantity<V, U>;

//     fn sub(self, rhs: V) -> Self::Output {
//         Self::Output {
//             value: self.value - rhs,
//             _phantom: PhantomData,
//         }
//     }
// }

// // mul

// impl<V, U> Mul<Quantity<V, U>> for Quantity<V, U>
// where
//     V: Mul<Output = V>,
//     U: Unit,
// {
//     type Output = Quantity<V, U>;

//     fn mul(self, rhs: Self) -> Self::Output {
//         Self::Output {
//             value: self.value * rhs.value,
//             _phantom: PhantomData,
//         }
//     }
// }

// impl<V, U> Mul<V> for Quantity<V, U>
// where
//     V: Mul<Output = V>,
//     U: Unit,
// {
//     type Output = Quantity<V, U>;

//     fn mul(self, rhs: V) -> Self::Output {
//         Self::Output {
//             value: self.value * rhs,
//             _phantom: PhantomData,
//         }
//     }
// }

// // div

// impl<V, U> Div<Quantity<V, U>> for Quantity<V, U>
// where
//     V: Div<Output = V>,
//     U: Unit,
// {
//     type Output = Quantity<V, U>;

//     fn div(self, rhs: Self) -> Self::Output {
//         Self::Output {
//             value: self.value / rhs.value,
//             _phantom: PhantomData,
//         }
//     }
// }

// impl<V, U> Div<V> for Quantity<V, U>
// where
//     V: Div<Output = V>,
//     U: Unit,
// {
//     type Output = Quantity<V, U>;

//     fn div(self, rhs: V) -> Self::Output {
//         Self::Output {
//             value: self.value / rhs,
//             _phantom: PhantomData,
//         }
//     }
// }

// // rem

// impl<V, U> Rem<Quantity<V, U>> for Quantity<V, U>
// where
//     V: Rem<Output = V>,
//     U: Unit,
// {
//     type Output = Quantity<V, U>;

//     fn rem(self, rhs: Self) -> Self::Output {
//         Self::Output {
//             value: self.value % rhs.value,
//             _phantom: PhantomData,
//         }
//     }
// }

// impl<V, U> Rem<V> for Quantity<V, U>
// where
//     V: Rem<Output = V>,
//     U: Unit,
// {
//     type Output = Quantity<V, U>;

//     fn rem(self, rhs: V) -> Self::Output {
//         Self::Output {
//             value: self.value % rhs,
//             _phantom: PhantomData,
//         }
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn unit_type_test() {
//         let pixels: Quantity<u32, Pixel> = 640.into();
//         assert_eq!(*pixels, 640);

//         let ratio: Quantity<f64, Ratio> = pixels.cast::<f64>().div_to(1000.0);
//         assert_eq!(*ratio, 640 as f64 / 1000.0);
//     }
// }
