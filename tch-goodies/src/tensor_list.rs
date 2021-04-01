use crate::common::*;

pub trait TensorList {
    fn into_owned_tensors(self) -> Vec<Tensor>;
}

impl TensorList for Vec<Tensor> {
    fn into_owned_tensors(self) -> Vec<Tensor> {
        self
    }
}

impl TensorList for Vec<&Tensor> {
    fn into_owned_tensors(self) -> Vec<Tensor> {
        let tensors: Vec<_> = self
            .into_iter()
            .map(|tensor| tensor.shallow_clone())
            .collect();
        tensors
    }
}

impl TensorList for &Vec<&Tensor> {
    fn into_owned_tensors(self) -> Vec<Tensor> {
        let tensors: Vec<_> = self
            .into_iter()
            .map(|&tensor| tensor.shallow_clone())
            .collect();
        tensors
    }
}

impl TensorList for &Vec<Tensor> {
    fn into_owned_tensors(self) -> Vec<Tensor> {
        let tensors: Vec<_> = self
            .into_iter()
            .map(|tensor| tensor.shallow_clone())
            .collect();
        tensors
    }
}

impl TensorList for &[Tensor] {
    fn into_owned_tensors(self) -> Vec<Tensor> {
        let tensors: Vec<_> = self.iter().map(|tensor| tensor.shallow_clone()).collect();
        tensors
    }
}

impl TensorList for &[&Tensor] {
    fn into_owned_tensors(self) -> Vec<Tensor> {
        let tensors: Vec<_> = self.iter().map(|&tensor| tensor.shallow_clone()).collect();
        tensors
    }
}

impl<const SIZE: usize> TensorList for &[&Tensor; SIZE] {
    fn into_owned_tensors(self) -> Vec<Tensor> {
        let tensors: Vec<_> = self.iter().map(|&tensor| tensor.shallow_clone()).collect();
        tensors
    }
}

impl<const SIZE: usize> TensorList for &[Tensor; SIZE] {
    fn into_owned_tensors(self) -> Vec<Tensor> {
        let tensors: Vec<_> = self.iter().map(|tensor| tensor.shallow_clone()).collect();
        tensors
    }
}

impl<const SIZE: usize> TensorList for [&Tensor; SIZE] {
    fn into_owned_tensors(self) -> Vec<Tensor> {
        let tensors: Vec<_> = self.iter().map(|&tensor| tensor.shallow_clone()).collect();
        tensors
    }
}

impl<const SIZE: usize> TensorList for [Tensor; SIZE] {
    fn into_owned_tensors(self) -> Vec<Tensor> {
        Vec::from(self)
    }
}
