use crate::common::*;
use model_config::config::Shape;

#[derive(Debug)]
pub struct Input {
    shape: Shape,
}

impl Input {
    pub fn new(shape: &Shape) -> Self {
        Self {
            shape: shape.clone(),
        }
    }

    pub fn forward(&self, tensor: &Tensor) -> Result<Tensor> {
        let actual_shape: Shape = tensor
            .size()
            .into_iter()
            .map(|size| size as usize)
            .collect();

        ensure!(
            actual_shape.is_compatible_with(&self.shape),
            "input shape mismatch: expect {}, but get {}",
            self.shape,
            actual_shape
        );

        Ok(tensor.shallow_clone())
    }
}
