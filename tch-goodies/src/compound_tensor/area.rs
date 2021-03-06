use crate::common::*;

/// Unchecked tensor of batched areas.
#[derive(Debug, TensorLike)]
pub struct AreaTensorUnchecked {
    pub area: Tensor,
}

/// Checked tensor of batched areas.
#[derive(Debug, TensorLike, Getters)]
pub struct AreaTensor {
    #[get = "pub"]
    pub(super) area: Tensor,
}

impl AreaTensor {
    pub fn num_samples(&self) -> i64 {
        let (num, _) = self.area.size2().unwrap();
        num
    }

    pub fn device(&self) -> Device {
        self.area.device()
    }
}

impl TryFrom<AreaTensorUnchecked> for AreaTensor {
    type Error = Error;

    fn try_from(from: AreaTensorUnchecked) -> Result<Self, Self::Error> {
        let AreaTensorUnchecked { area } = from;
        match area.size2()? {
            (_, 1) => (),
            _ => bail!("size_mismatch"),
        }
        Ok(Self { area })
    }
}

impl From<AreaTensor> for AreaTensorUnchecked {
    fn from(from: AreaTensor) -> Self {
        let AreaTensor { area } = from;
        Self { area }
    }
}
