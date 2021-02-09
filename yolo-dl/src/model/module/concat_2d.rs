use super::*;

#[derive(Debug)]
pub struct Concat2D;

impl Concat2D {
    pub fn forward<T>(&self, tensors: &[T]) -> Result<Tensor>
    where
        T: Borrow<Tensor>,
    {
        tensors.iter().try_for_each(|tensor| -> Result<_> {
            tensor.borrow().size4()?;
            Ok(())
        })?;
        let output = Tensor::f_cat(tensors, 1)?;
        Ok(output)
    }
}
