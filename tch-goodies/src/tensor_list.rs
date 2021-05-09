use crate::common::*;

pub use list::*;
pub use optional_list::*;

mod list {
    use super::*;

    pub const EMPTY_TENSORS: &[Tensor] = &[];

    pub trait TensorList {
        fn into_tensor_list(self) -> Vec<Tensor>;
    }

    impl TensorList for Vec<Tensor> {
        fn into_tensor_list(self) -> Vec<Tensor> {
            self
        }
    }

    impl TensorList for Vec<&Tensor> {
        fn into_tensor_list(self) -> Vec<Tensor> {
            let tensors: Vec<_> = self
                .into_iter()
                .map(|tensor| tensor.shallow_clone())
                .collect();
            tensors
        }
    }

    impl TensorList for &Vec<&Tensor> {
        fn into_tensor_list(self) -> Vec<Tensor> {
            let tensors: Vec<_> = self.iter().map(|&tensor| tensor.shallow_clone()).collect();
            tensors
        }
    }

    impl TensorList for &Vec<Tensor> {
        fn into_tensor_list(self) -> Vec<Tensor> {
            let tensors: Vec<_> = self.iter().map(|tensor| tensor.shallow_clone()).collect();
            tensors
        }
    }

    impl TensorList for &[Tensor] {
        fn into_tensor_list(self) -> Vec<Tensor> {
            let tensors: Vec<_> = self.iter().map(|tensor| tensor.shallow_clone()).collect();
            tensors
        }
    }

    impl TensorList for &[&Tensor] {
        fn into_tensor_list(self) -> Vec<Tensor> {
            let tensors: Vec<_> = self.iter().map(|&tensor| tensor.shallow_clone()).collect();
            tensors
        }
    }

    impl<const SIZE: usize> TensorList for &[&Tensor; SIZE] {
        fn into_tensor_list(self) -> Vec<Tensor> {
            let tensors: Vec<_> = self.iter().map(|&tensor| tensor.shallow_clone()).collect();
            tensors
        }
    }

    impl<const SIZE: usize> TensorList for &[Tensor; SIZE] {
        fn into_tensor_list(self) -> Vec<Tensor> {
            let tensors: Vec<_> = self.iter().map(|tensor| tensor.shallow_clone()).collect();
            tensors
        }
    }

    impl<const SIZE: usize> TensorList for [&Tensor; SIZE] {
        fn into_tensor_list(self) -> Vec<Tensor> {
            let tensors: Vec<_> = self.iter().map(|&tensor| tensor.shallow_clone()).collect();
            tensors
        }
    }

    impl<const SIZE: usize> TensorList for [Tensor; SIZE] {
        fn into_tensor_list(self) -> Vec<Tensor> {
            Vec::from(self)
        }
    }
}

mod optional_list {
    use super::*;

    pub const NONE_TENSORS: Option<&[Tensor]> = None;

    pub trait OptionalTensorList {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>>;
    }

    impl OptionalTensorList for Vec<Tensor> {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            Some(self)
        }
    }

    impl OptionalTensorList for Vec<&Tensor> {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            let tensors: Vec<_> = self
                .into_iter()
                .map(|tensor| tensor.shallow_clone())
                .collect();
            Some(tensors)
        }
    }

    impl OptionalTensorList for &Vec<&Tensor> {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            let tensors: Vec<_> = self.iter().map(|&tensor| tensor.shallow_clone()).collect();
            Some(tensors)
        }
    }

    impl OptionalTensorList for &Vec<Tensor> {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            let tensors: Vec<_> = self.iter().map(|tensor| tensor.shallow_clone()).collect();
            Some(tensors)
        }
    }

    impl OptionalTensorList for &[Tensor] {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            let tensors: Vec<_> = self.iter().map(|tensor| tensor.shallow_clone()).collect();
            Some(tensors)
        }
    }

    impl OptionalTensorList for &[&Tensor] {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            let tensors: Vec<_> = self.iter().map(|&tensor| tensor.shallow_clone()).collect();
            Some(tensors)
        }
    }

    impl<const SIZE: usize> OptionalTensorList for &[&Tensor; SIZE] {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            let tensors: Vec<_> = self.iter().map(|&tensor| tensor.shallow_clone()).collect();
            Some(tensors)
        }
    }

    impl<const SIZE: usize> OptionalTensorList for &[Tensor; SIZE] {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            let tensors: Vec<_> = self.iter().map(|tensor| tensor.shallow_clone()).collect();
            Some(tensors)
        }
    }

    impl<const SIZE: usize> OptionalTensorList for [&Tensor; SIZE] {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            let tensors: Vec<_> = self.iter().map(|&tensor| tensor.shallow_clone()).collect();
            Some(tensors)
        }
    }

    impl<const SIZE: usize> OptionalTensorList for [Tensor; SIZE] {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            Some(Vec::from(self))
        }
    }

    impl OptionalTensorList for Option<Vec<Tensor>> {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            self
        }
    }

    impl OptionalTensorList for Option<Vec<&Tensor>> {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            self.map(|tensors| tensors.into_tensor_list())
        }
    }

    impl OptionalTensorList for Option<&Vec<&Tensor>> {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            self.map(|tensors| tensors.into_tensor_list())
        }
    }

    impl OptionalTensorList for Option<&Vec<Tensor>> {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            self.map(|tensors| tensors.into_tensor_list())
        }
    }

    impl OptionalTensorList for Option<&[Tensor]> {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            self.map(|tensors| tensors.into_tensor_list())
        }
    }

    impl OptionalTensorList for Option<&[&Tensor]> {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            self.map(|tensors| tensors.into_tensor_list())
        }
    }

    impl<const SIZE: usize> OptionalTensorList for Option<&[&Tensor; SIZE]> {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            self.map(|tensors| tensors.into_tensor_list())
        }
    }

    impl<const SIZE: usize> OptionalTensorList for Option<&[Tensor; SIZE]> {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            self.map(|tensors| tensors.into_tensor_list())
        }
    }

    impl<const SIZE: usize> OptionalTensorList for Option<[&Tensor; SIZE]> {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            self.map(|tensors| tensors.into_tensor_list())
        }
    }

    impl<const SIZE: usize> OptionalTensorList for Option<[Tensor; SIZE]> {
        fn into_optional_tensor_list(self) -> Option<Vec<Tensor>> {
            self.map(|tensors| tensors.into_tensor_list())
        }
    }
}
