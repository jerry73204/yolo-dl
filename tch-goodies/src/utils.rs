use crate::{common::*, ratio::Ratio};

pub use into_tch_element::*;

pub const EPSILON: f64 = 1e-16;

mod into_tch_element {
    use super::*;

    pub trait IntoTchElement
    where
        Self::Output: Element,
    {
        type Output;

        fn into_tch_element(self) -> Self::Output;
    }

    impl IntoTchElement for u8 {
        type Output = u8;

        fn into_tch_element(self) -> Self::Output {
            self
        }
    }

    impl IntoTchElement for i8 {
        type Output = i8;

        fn into_tch_element(self) -> Self::Output {
            self
        }
    }

    impl IntoTchElement for i16 {
        type Output = i16;

        fn into_tch_element(self) -> Self::Output {
            self
        }
    }

    impl IntoTchElement for bool {
        type Output = bool;

        fn into_tch_element(self) -> Self::Output {
            self
        }
    }

    impl IntoTchElement for f32 {
        type Output = f32;
        fn into_tch_element(self) -> Self::Output {
            self
        }
    }

    impl IntoTchElement for f64 {
        type Output = f64;
        fn into_tch_element(self) -> Self::Output {
            self
        }
    }

    impl IntoTchElement for R64 {
        type Output = f64;

        fn into_tch_element(self) -> Self::Output {
            self.raw()
        }
    }

    impl IntoTchElement for R32 {
        type Output = f32;

        fn into_tch_element(self) -> Self::Output {
            self.raw()
        }
    }

    impl IntoTchElement for Ratio {
        type Output = f64;

        fn into_tch_element(self) -> Self::Output {
            self.to_f64()
        }
    }
}
