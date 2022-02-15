#[macro_export]
macro_rules! unit_wrapper {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name<T>(pub T);

        impl<T> From<T> for $name<T> {
            fn from(value: T) -> Self {
                Self(value)
            }
        }

        impl<L, R> std::ops::Add<$name<R>> for $name<L>
        where
            L: std::ops::Add<R>,
        {
            type Output = $name<<L as std::ops::Add<R>>::Output>;

            fn add(self, rhs: $name<R>) -> Self::Output {
                use std::ops::Add;
                $name(self.0.add(rhs.0))
            }
        }

        impl<L, R> std::ops::Sub<$name<R>> for $name<L>
        where
            L: std::ops::Sub<R>,
        {
            type Output = $name<<L as std::ops::Sub<R>>::Output>;

            fn sub(self, rhs: $name<R>) -> Self::Output {
                use std::ops::Sub;
                $name(self.0.sub(rhs.0))
            }
        }

        impl<L, R> std::ops::Mul<$name<R>> for $name<L>
        where
            L: std::ops::Mul<R>,
        {
            type Output = $name<<L as std::ops::Mul<R>>::Output>;

            fn mul(self, rhs: $name<R>) -> Self::Output {
                use std::ops::Mul;
                $name(self.0.mul(rhs.0))
            }
        }

        impl<L, R> std::ops::Div<$name<R>> for $name<L>
        where
            L: std::ops::Div<R>,
        {
            type Output = $name<<L as std::ops::Div<R>>::Output>;

            fn div(self, rhs: $name<R>) -> Self::Output {
                use std::ops::Div;
                $name(self.0.div(rhs.0))
            }
        }
    };
}
