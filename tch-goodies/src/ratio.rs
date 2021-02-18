use crate::common::*;

/// A floating point wrapper that restricts its range from 0 to 1.
#[derive(Debug, Clone, Copy, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Ratio(R64);

impl Zero for Ratio {
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    fn zero() -> Self {
        Self(R64::zero())
    }
}

impl One for Ratio {
    fn one() -> Self {
        Self(R64::one())
    }
}

impl Num for Ratio {
    type FromStrRadixErr = Error;

    fn from_str_radix(text: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let value: Self = f64::from_str_radix(text, radix)
            .map_err(|err| format_err!("{:?}", err))?
            .try_into()?;
        Ok(value)
    }
}

impl Neg for Ratio {
    type Output = Self;

    fn neg(self) -> Self::Output {
        panic!("negation is not supported");
    }
}

impl ToPrimitive for Ratio {
    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }
}

impl NumCast for Ratio {
    fn from<T: ToPrimitive>(from: T) -> Option<Self> {
        let value = <f64 as NumCast>::from(from)?;
        (0.0..=1.0).contains(&value).then(|| Self(r64(value)))
    }
}

impl Float for Ratio {
    fn is_nan(self) -> bool {
        panic!("unsupported operation");
    }

    fn nan() -> Self {
        panic!("unsupported operation");
    }

    fn infinity() -> Self {
        panic!("unsupported operation");
    }

    fn neg_infinity() -> Self {
        panic!("unsupported operation");
    }

    fn neg_zero() -> Self {
        panic!("unsupported operation");
    }

    fn min_value() -> Self {
        panic!("unsupported operation");
    }

    fn min_positive_value() -> Self {
        panic!("unsupported operation");
    }

    fn max_value() -> Self {
        panic!("unsupported operation");
    }

    fn is_infinite(self) -> bool {
        panic!("unsupported operation");
    }

    fn is_finite(self) -> bool {
        panic!("unsupported operation");
    }

    fn is_normal(self) -> bool {
        panic!("unsupported operation");
    }

    fn classify(self) -> FpCategory {
        panic!("unsupported operation");
    }

    fn floor(self) -> Self {
        panic!("unsupported operation");
    }

    fn ceil(self) -> Self {
        panic!("unsupported operation");
    }

    fn round(self) -> Self {
        panic!("unsupported operation");
    }

    fn trunc(self) -> Self {
        panic!("unsupported operation");
    }

    fn fract(self) -> Self {
        panic!("unsupported operation");
    }

    fn abs(self) -> Self {
        panic!("unsupported operation");
    }

    fn signum(self) -> Self {
        panic!("unsupported operation");
    }

    fn is_sign_positive(self) -> bool {
        panic!("unsupported operation");
    }

    fn is_sign_negative(self) -> bool {
        panic!("unsupported operation");
    }

    fn mul_add(self, _a: Self, _b: Self) -> Self {
        panic!("unsupported operation");
    }

    fn recip(self) -> Self {
        panic!("unsupported operation");
    }

    fn powi(self, _n: i32) -> Self {
        panic!("unsupported operation");
    }

    fn powf(self, _n: Self) -> Self {
        panic!("unsupported operation");
    }

    fn sqrt(self) -> Self {
        panic!("unsupported operation");
    }

    fn exp(self) -> Self {
        panic!("unsupported operation");
    }

    fn exp2(self) -> Self {
        panic!("unsupported operation");
    }

    fn ln(self) -> Self {
        panic!("unsupported operation");
    }

    fn log(self, _base: Self) -> Self {
        panic!("unsupported operation");
    }

    fn log2(self) -> Self {
        panic!("unsupported operation");
    }

    fn log10(self) -> Self {
        panic!("unsupported operation");
    }

    fn max(self, other: Self) -> Self {
        Ratio(self.0.max(other.0))
    }

    fn min(self, other: Self) -> Self {
        Ratio(self.0.min(other.0))
    }

    fn abs_sub(self, _other: Self) -> Self {
        panic!("unsupported operation");
    }

    fn cbrt(self) -> Self {
        panic!("unsupported operation");
    }

    fn hypot(self, _other: Self) -> Self {
        panic!("unsupported operation");
    }

    fn sin(self) -> Self {
        panic!("unsupported operation");
    }

    fn cos(self) -> Self {
        panic!("unsupported operation");
    }

    fn tan(self) -> Self {
        panic!("unsupported operation");
    }

    fn asin(self) -> Self {
        panic!("unsupported operation");
    }

    fn acos(self) -> Self {
        panic!("unsupported operation");
    }

    fn atan(self) -> Self {
        panic!("unsupported operation");
    }

    fn atan2(self, _other: Self) -> Self {
        panic!("unsupported operation");
    }

    fn sin_cos(self) -> (Self, Self) {
        panic!("unsupported operation");
    }

    fn exp_m1(self) -> Self {
        panic!("unsupported operation");
    }

    fn ln_1p(self) -> Self {
        panic!("unsupported operation");
    }

    fn sinh(self) -> Self {
        panic!("unsupported operation");
    }

    fn cosh(self) -> Self {
        panic!("unsupported operation");
    }

    fn tanh(self) -> Self {
        panic!("unsupported operation");
    }

    fn asinh(self) -> Self {
        panic!("unsupported operation");
    }

    fn acosh(self) -> Self {
        panic!("unsupported operation");
    }

    fn atanh(self) -> Self {
        panic!("unsupported operation");
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        panic!("unsupported operation");
    }
}

impl Ratio {
    pub fn to_r64(&self) -> R64 {
        self.0
    }

    pub fn to_f64(&self) -> f64 {
        self.0.raw()
    }

    pub fn checked_add(&self, rhs: Ratio) -> Result<Self> {
        Ratio::try_from(self.0 + rhs.0)
    }

    pub fn checked_sub(&self, rhs: Ratio) -> Result<Self> {
        Ratio::try_from(self.0 - rhs.0)
    }

    pub fn checked_mul(&self, rhs: Ratio) -> Result<Self> {
        Ratio::try_from(self.0 * rhs.0)
    }

    pub fn checked_div(&self, rhs: Ratio) -> Result<Self> {
        Ratio::try_from(self.0 / rhs.0)
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

impl Rem<Ratio> for Ratio {
    type Output = Ratio;

    fn rem(self, rhs: Ratio) -> Self::Output {
        Self::try_from(self.0 % rhs.0).unwrap()
    }
}

impl Rem<f64> for Ratio {
    type Output = Ratio;

    fn rem(self, rhs: f64) -> Self::Output {
        Self::try_from(self.0 % rhs).unwrap()
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

impl PartialOrd<R64> for Ratio {
    fn partial_cmp(&self, other: &R64) -> Option<Ordering> {
        self.0.partial_cmp(other)
    }
}

impl PartialOrd<f64> for Ratio {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.0.raw().partial_cmp(other)
    }
}

impl Display for Ratio {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        self.to_f64().fmt(f)
    }
}
