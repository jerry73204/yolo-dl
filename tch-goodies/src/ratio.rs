use crate::common::*;

#[derive(Debug, Clone, Copy, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Ratio(R64);

impl Ratio {
    pub fn to_r64(&self) -> R64 {
        self.0
    }

    pub fn to_f64(&self) -> f64 {
        self.0.raw()
    }

    pub fn checked_add(&self, rhs: Ratio) -> Result<Self> {
        Ok(Ratio::try_from(self.0 + rhs.0)?)
    }

    pub fn checked_sub(&self, rhs: Ratio) -> Result<Self> {
        Ok(Ratio::try_from(self.0 - rhs.0)?)
    }

    pub fn checked_mul(&self, rhs: Ratio) -> Result<Self> {
        Ok(Ratio::try_from(self.0 * rhs.0)?)
    }

    pub fn checked_div(&self, rhs: Ratio) -> Result<Self> {
        Ok(Ratio::try_from(self.0 / rhs.0)?)
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
        Self::try_from(value).map_err(|err| D::Error::custom(format!("{:?}", err)))
    }
}

impl AbsDiffEq<Ratio> for Ratio {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Ratio, epsilon: Self::Epsilon) -> bool {
        self.0.raw().abs_diff_eq(&other.0.raw(), epsilon)
    }
}

impl AbsDiffEq<f64> for Ratio {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &f64, epsilon: Self::Epsilon) -> bool {
        self.0.raw().abs_diff_eq(other, epsilon)
    }
}

impl TryFrom<R64> for Ratio {
    type Error = Error;

    fn try_from(value: R64) -> Result<Self, Self::Error> {
        ensure!(
            ((0.0 - f64::default_epsilon())..=(1.0 + f64::default_epsilon()))
                .contains(&value.raw()),
            "ratio value must be within range [0.0, 1.0], but get {}",
            value
        );
        let value = value.max(R64::new(0.0)).min(R64::new(1.0));
        Ok(Self(value))
    }
}

impl TryFrom<f64> for Ratio {
    type Error = Error;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::try_from(R64::try_new(value).ok_or_else(|| format_err!("not a finite value"))?)
    }
}

impl TryFrom<f32> for Ratio {
    type Error = Error;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::try_from(R64::try_new(value as f64).ok_or_else(|| format_err!("not a finite value"))?)
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

// impl Deref for Ratio {
//     type Target = R64;

//     fn deref(&self) -> &Self::Target {
//         &self.0
//     }
// }

impl Add<Ratio> for Ratio {
    type Output = Ratio;

    fn add(self, rhs: Ratio) -> Self::Output {
        self.checked_add(rhs).unwrap()
    }
}

// impl Add<f64> for Ratio {
//     type Output = Ratio;

//     fn add(self, rhs: f64) -> Self::Output {
//         self.checked_add(Ratio::try_from(rhs).unwrap()).unwrap()
//     }
// }

impl Sub<Ratio> for Ratio {
    type Output = Ratio;

    fn sub(self, rhs: Ratio) -> Self::Output {
        self.checked_sub(rhs).unwrap()
    }
}

// impl Sub<f64> for Ratio {
//     type Output = Ratio;

//     fn sub(self, rhs: f64) -> Self::Output {
//         self.checked_sub(Ratio::try_from(rhs).unwrap()).unwrap()
//     }
// }

impl Mul<Ratio> for Ratio {
    type Output = Ratio;

    fn mul(self, rhs: Ratio) -> Self::Output {
        self.checked_mul(rhs).unwrap()
    }
}

impl Mul<f64> for Ratio {
    type Output = Ratio;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::try_from(self.0 * rhs).unwrap()
    }
}

impl Div<Ratio> for Ratio {
    type Output = Ratio;

    fn div(self, rhs: Ratio) -> Self::Output {
        self.checked_div(rhs).unwrap()
    }
}

impl Div<f64> for Ratio {
    type Output = Ratio;

    fn div(self, rhs: f64) -> Self::Output {
        Self::try_from(self.0 / rhs).unwrap()
    }
}

impl PartialEq<Ratio> for Ratio {
    fn eq(&self, rhs: &Ratio) -> bool {
        self.0.eq(&rhs.0)
    }
}

impl PartialEq<R64> for Ratio {
    fn eq(&self, rhs: &R64) -> bool {
        self.0.eq(rhs)
    }
}

impl PartialEq<f64> for Ratio {
    fn eq(&self, rhs: &f64) -> bool {
        self.0.raw().eq(rhs)
    }
}

impl Display for Ratio {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        self.to_f64().fmt(f)
    }
}
