use crate::common::*;

#[derive(Debug)]
pub struct Concat2D {
    _private: [u8; 0],
}

impl Concat2D {
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
        let output = Tensor::f_cat(&tensors, 1)?;
        Ok(output)
    }
}
