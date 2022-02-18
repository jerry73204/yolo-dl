#[macro_export]
macro_rules! unit_wrapper {
    ($name:ident) => { unit_wrapper!(() $name); };
    (pub $name:ident) => { unit_wrapper!((pub) $name); };
    (($($vis:tt)*) $name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        $($vis)* struct $name<T>(pub T);

        impl<T> $name<T> {
            pub fn as_ref(&self) -> Pixel<&T> {
                Pixel(&self.0)
            }

            pub fn as_mut(&mut self) -> Pixel<&mut T> {
                Pixel(&mut self.0)
            }

            pub fn as_deref(&self) -> Pixel<&<T as std::ops::Deref>::Target>
            where
                T: std::ops::Deref
            {
                use std::ops::Deref;
                Pixel(self.deref())
            }

            pub fn as_deref_mut(&mut self) -> Pixel<&mut <T as std::ops::Deref>::Target>
            where
                T: std::ops::DerefMut
            {
                use std::ops::DerefMut;
                Pixel(self.deref_mut())
            }
        }

        impl<T> From<T> for $name<T> {
            fn from(value: T) -> Self {
                Self(value)
            }
        }

        impl<T> std::ops::Deref for $name<T> {
            type Target = T;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<T> std::ops::DerefMut for $name<T> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
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

        impl<'a, L, R> std::ops::Add<&'a $name<R>> for &'a $name<L>
        where
            L: 'a,
            R: 'a,
            &'a L: std::ops::Add<&'a R>,
        {
            type Output = $name<<&'a L as std::ops::Add<&'a R>>::Output>;

            fn add(self, rhs: &'a $name<R>) -> Self::Output {
                use std::ops::Add;
                $name((&self.0).add(&rhs.0))
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


        impl<'a, L, R> std::ops::Sub<&'a $name<R>> for &'a $name<L>
        where
            L: 'a,
            R: 'a,
            &'a L: std::ops::Sub<&'a R>,
        {
            type Output = $name<<&'a L as std::ops::Sub<&'a R>>::Output>;

            fn sub(self, rhs: &'a $name<R>) -> Self::Output {
                use std::ops::Sub;
                $name((&self.0).sub(&rhs.0))
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


        impl<'a, L, R> std::ops::Mul<&'a $name<R>> for &'a $name<L>
        where
            L: 'a,
            R: 'a,
            &'a L: std::ops::Mul<&'a R>,
        {
            type Output = $name<<&'a L as std::ops::Mul<&'a R>>::Output>;

            fn mul(self, rhs: &'a $name<R>) -> Self::Output {
                use std::ops::Mul;
                $name((&self.0).mul(&rhs.0))
            }
        }


        impl<L, R> std::ops::Rem<$name<R>> for $name<L>
        where
            L: std::ops::Rem<R>,
        {
            type Output = $name<<L as std::ops::Rem<R>>::Output>;

            fn rem(self, rhs: $name<R>) -> Self::Output {
                use std::ops::Rem;
                $name(self.0.rem(rhs.0))
            }
        }

        impl<'a, L, R> std::ops::Rem<&'a $name<R>> for &'a $name<L>
        where
            L: 'a,
            R: 'a,
            &'a L: std::ops::Rem<&'a R>,
        {
            type Output = $name<<&'a L as std::ops::Rem<&'a R>>::Output>;

            fn rem(self, rhs: &'a $name<R>) -> Self::Output {
                use std::ops::Rem;
                $name((&self.0).rem(&rhs.0))
            }
        }
    };
}
