use super::*;

#[derive(Debug)]
pub struct Sum2D;

impl Sum2D {
    pub fn forward<T>(&self, tensors: &[T]) -> Result<Tensor>
    where
        T: Borrow<Tensor>,
    {
        tensors.iter().try_for_each(|tensor| -> Result<_> {
            tensor.borrow().size4()?;
            Ok(())
        })?;
        let mut iter = tensors.iter();
        let first = iter
            .next()
            .ok_or_else(|| format_err!("empty input is not allowed"))?
            .borrow()
            .shallow_clone();
        let output = iter.try_fold(first, |acc, tensor| acc.f_add(tensor.borrow()))?;
        Ok(output)
    }
}
