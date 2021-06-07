use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, TensorLike)]
pub struct FlatIndex {
    pub batch_index: i64,
    pub flat_index: i64,
}

#[derive(Debug, TensorLike)]
pub struct FlatIndexTensor {
    pub batches: Tensor,
    pub flats: Tensor,
}

impl FlatIndexTensor {
    pub fn num_samples(&self) -> i64 {
        self.batches.size1().unwrap()
    }

    pub fn cat_mini_batches<T>(indexes: impl IntoIterator<Item = (T, i64)>) -> Self
    where
        T: Borrow<Self>,
    {
        let (batches_vec, flats_vec) = indexes
            .into_iter()
            .scan(0i64, |batch_base, (flat_indexes, mini_batch_size)| {
                let Self { batches, flats } = flat_indexes.borrow().shallow_clone();

                let new_batches = batches + *batch_base;
                *batch_base += mini_batch_size;

                Some((new_batches, flats))
            })
            .unzip_n_vec();

        Self {
            batches: Tensor::cat(&batches_vec, 0),
            flats: Tensor::cat(&flats_vec, 0),
        }
    }
}

impl<'a> From<&'a FlatIndexTensor> for Vec<FlatIndex> {
    fn from(from: &'a FlatIndexTensor) -> Self {
        let FlatIndexTensor { batches, flats } = from;

        izip!(Vec::<i64>::from(batches), Vec::<i64>::from(flats))
            .map(|(batch_index, flat_index)| FlatIndex {
                batch_index,
                flat_index,
            })
            .collect()
    }
}

impl From<FlatIndexTensor> for Vec<FlatIndex> {
    fn from(from: FlatIndexTensor) -> Self {
        (&from).into()
    }
}

impl FromIterator<FlatIndex> for FlatIndexTensor {
    fn from_iter<T: IntoIterator<Item = FlatIndex>>(iter: T) -> Self {
        let (batches, flats) = iter
            .into_iter()
            .map(|index| {
                let FlatIndex {
                    batch_index,
                    flat_index,
                } = index;
                (batch_index, flat_index)
            })
            .unzip_n_vec();
        Self {
            batches: Tensor::of_slice(&batches),
            flats: Tensor::of_slice(&flats),
        }
    }
}

impl<'a> FromIterator<&'a FlatIndex> for FlatIndexTensor {
    fn from_iter<T: IntoIterator<Item = &'a FlatIndex>>(iter: T) -> Self {
        let (batches, flats) = iter
            .into_iter()
            .map(|index| {
                let FlatIndex {
                    batch_index,
                    flat_index,
                } = *index;
                (batch_index, flat_index)
            })
            .unzip_n_vec();
        Self {
            batches: Tensor::of_slice(&batches),
            flats: Tensor::of_slice(&flats),
        }
    }
}
