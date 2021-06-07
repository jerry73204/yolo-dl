use super::DenseDetectionTensor;
use crate::common::*;

#[derive(Debug, TensorLike)]
pub struct DenseDetectionTensorList {
    pub(super) inner: DenseDetectionTensorListUnchecked,
}

impl DenseDetectionTensorList {
    pub fn from_detection_tensors(
        tensors: impl IntoIterator<Item = impl Borrow<DenseDetectionTensor>>,
    ) -> Result<Self> {
        let (tensors, batch_size_set): (Vec<_>, HashSet<_>) = tensors
            .into_iter()
            .map(|tensor| {
                let tensor = tensor.borrow().shallow_clone();
                let batch_size = tensor.batch_size();
                (tensor, batch_size)
            })
            .unzip();

        ensure!(!tensors.is_empty());
        ensure!(batch_size_set.len() == 1);

        Ok(Self {
            inner: DenseDetectionTensorListUnchecked { tensors },
        })
    }

    pub fn batch_size(&self) -> usize {
        self.tensors[0].batch_size()
    }

    pub fn cat_batch(tensors: impl IntoIterator<Item = impl Borrow<Self>>) -> Result<Self> {
        Self::cat(tensors, 0)
    }

    pub fn cat_height(tensors: impl IntoIterator<Item = impl Borrow<Self>>) -> Result<Self> {
        Self::cat(tensors, 3)
    }

    pub fn cat_width(tensors: impl IntoIterator<Item = impl Borrow<Self>>) -> Result<Self> {
        Self::cat(tensors, 4)
    }

    fn cat(lists: impl IntoIterator<Item = impl Borrow<Self>>, index: i64) -> Result<Self> {
        // list index -> layer index -> tensor
        let tensors_vec: Vec<Vec<_>> = lists
            .into_iter()
            .map(|list| list.borrow().tensors.shallow_clone())
            .collect();

        // layer index -> list index -> tensor
        let tensors_vec = tensors_vec.transpose().unwrap();

        // concatenate each layer of tensors
        let tensors: Vec<_> = tensors_vec
            .into_iter()
            .map(|layer| DenseDetectionTensor::cat(layer, index))
            .try_collect()?;

        Ok(Self {
            inner: DenseDetectionTensorListUnchecked { tensors },
        })
    }
}

#[derive(Debug, TensorLike)]
pub struct DenseDetectionTensorListUnchecked {
    pub tensors: Vec<DenseDetectionTensor>,
}

impl Borrow<DenseDetectionTensorListUnchecked> for DenseDetectionTensorList {
    fn borrow(&self) -> &DenseDetectionTensorListUnchecked {
        &self.inner
    }
}

impl Deref for DenseDetectionTensorList {
    type Target = DenseDetectionTensorListUnchecked;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl From<DenseDetectionTensorList> for DenseDetectionTensorListUnchecked {
    fn from(from: DenseDetectionTensorList) -> Self {
        from.inner
    }
}

impl TryFrom<DenseDetectionTensorListUnchecked> for DenseDetectionTensorList {
    type Error = Error;

    fn try_from(from: DenseDetectionTensorListUnchecked) -> Result<Self, Self::Error> {
        let DenseDetectionTensorListUnchecked { tensors } = &from;

        let (batch_size_set, num_classes_set): (HashSet<_>, HashSet<_>) = tensors
            .iter()
            .map(|tensor| (tensor.batch_size(), tensor.num_classes()))
            .unzip();

        ensure!(num_classes_set.len() == 1);
        ensure!(batch_size_set.len() == 1);

        Ok(Self { inner: from })
    }
}
