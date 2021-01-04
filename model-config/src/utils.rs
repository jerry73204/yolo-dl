use crate::common::*;

pub use iterator_ex::*;
pub use option_ex::*;

pub fn hash_vec_indexset<T, H>(set: &IndexSet<T>, state: &mut H)
where
    T: Hash,
    H: Hasher,
{
    let set: Vec<_> = set.iter().collect();
    set.hash(state);
}

pub fn hash_vec_indexmap<K, V, H>(set: &IndexMap<K, V>, state: &mut H)
where
    K: Hash,
    V: Hash,
    H: Hasher,
{
    let map: Vec<_> = set.iter().collect();
    map.hash(state);
}

// pub fn empty_vec<T>() -> Vec<T> {
//     vec![]
// }

pub fn empty_index_set<T>() -> IndexSet<T> {
    IndexSet::new()
}

pub fn empty_index_map<K, V>() -> IndexMap<K, V> {
    IndexMap::new()
}

mod option_ex {
    use super::*;

    pub trait OptionEx<T> {
        fn display(&self) -> OptionDisplay<'_, T>
        where
            T: Display;
    }

    impl<T> OptionEx<T> for Option<T> {
        fn display(&self) -> OptionDisplay<'_, T>
        where
            T: Display,
        {
            OptionDisplay(self.as_ref())
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct OptionDisplay<'a, T>(Option<&'a T>);

    impl<T> Display for OptionDisplay<'_, T>
    where
        T: Display,
    {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            if let Self(Some(item)) = self {
                Display::fmt(item, f)?;
            }
            Ok(())
        }
    }
}

mod iterator_ex {
    use super::*;

    pub trait IteratorEx: Iterator {
        fn try_flat_map<F, U, In, Out, Error>(self, f: F) -> TryFlatMap<Self, F, U>
        where
            Self: Sized + Iterator<Item = Result<In, Error>>,
            F: FnMut(In) -> Result<U, Error>,
            U: IntoIterator<Item = Result<Out, Error>>,
        {
            TryFlatMap::Ready { outer: self, f }
        }
    }

    impl<T: Iterator> IteratorEx for T {}

    #[derive(Debug, Clone)]
    pub enum TryFlatMap<I, F, U>
    where
        U: IntoIterator,
    {
        Ready { outer: I, f: F },
        Unfolding { outer: I, inner: U::IntoIter, f: F },
        Finished,
    }

    impl<I, F, U, In, Out, Error> TryFlatMap<I, F, U>
    where
        I: Iterator<Item = Result<In, Error>>,
        F: FnMut(In) -> Result<U, Error>,
        U: IntoIterator<Item = Result<Out, Error>>,
    {
        fn into_finished(&mut self) {
            match self {
                Self::Finished => unreachable!(),
                _ => *self = Self::Finished,
            }
        }

        fn into_ready(&mut self) {
            take_mut::take(self, |state| match state {
                Self::Unfolding { outer, f, .. } => Self::Ready { outer, f },
                _ => unreachable!(),
            })
        }

        fn into_unfolding(&mut self, inner: U::IntoIter) {
            take_mut::take(self, |state| match state {
                Self::Ready { outer, f } => Self::Unfolding { outer, inner, f },
                _ => unreachable!(),
            })
        }
    }

    impl<I, F, U, In, Out, Error> Iterator for TryFlatMap<I, F, U>
    where
        I: Iterator<Item = Result<In, Error>>,
        F: FnMut(In) -> Result<U, Error>,
        U: IntoIterator<Item = Result<Out, Error>>,
    {
        type Item = Result<Out, Error>;

        fn next(&mut self) -> Option<Self::Item> {
            loop {
                match self {
                    Self::Finished => break None,
                    Self::Ready { outer, f } => match outer.next() {
                        Some(Ok(in_item)) => {
                            let mut inner = match f(in_item) {
                                Ok(inner) => inner.into_iter(),
                                Err(err) => {
                                    self.into_finished();
                                    break Some(Err(err));
                                }
                            };

                            match inner.next() {
                                Some(Ok(out_item)) => {
                                    self.into_unfolding(inner);
                                    break Some(Ok(out_item));
                                }
                                Some(Err(err)) => {
                                    self.into_finished();
                                    break Some(Err(err));
                                }
                                None => {}
                            }
                        }
                        Some(Err(err)) => {
                            self.into_finished();
                            break Some(Err(err));
                        }
                        None => {
                            self.into_finished();
                            break None;
                        }
                    },
                    Self::Unfolding { inner, .. } => match inner.next() {
                        Some(Ok(item)) => break Some(Ok(item)),
                        Some(Err(err)) => {
                            self.into_finished();
                            break Some(Err(err));
                        }
                        None => {
                            self.into_ready();
                        }
                    },
                }
            }
        }
    }
}
