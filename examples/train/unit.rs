use crate::common::*;

pub use ratio::*;

// ratio wrapper

mod ratio {
    use super::*;

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
}
