use crate::common::*;

#[derive(Debug)]
pub struct Sum2D {
    _private: [u8; 0],
}

impl Sum2D {
    pub fn new() -> Self {
        Self { _private: [] }
    }

    pub fn forward(
        &self,
        tensors: impl IntoIterator<Item = impl Borrow<Tensor>>,
    ) -> Result<Tensor> {
        let tensors: Vec<_> = tensors
            .into_iter()
            .map(|tensor| -> Result<_> {
                tensor.borrow().size4()?;
                Ok(tensor)
            })
            .try_collect()?;
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
