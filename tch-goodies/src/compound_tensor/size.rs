use crate::common::*;

/// Unchecked tensor of batched sizes.
#[derive(Debug, TensorLike)]
pub struct SizeTensorUnchecked {
    pub h: Tensor,
    pub w: Tensor,
}

/// Checked tensor of batched sizes.
#[derive(Debug, TensorLike, Getters)]
pub struct SizeTensor {
    #[get = "pub"]
    pub(super) h: Tensor,
    #[get = "pub"]
    pub(super) w: Tensor,
}

impl SizeTensor {
    pub fn num_samples(&self) -> i64 {
        let (num, _) = self.h.size2().unwrap();
        num
    }

    pub fn device(&self) -> Device {
        self.h.device()
    }
}

impl TryFrom<SizeTensorUnchecked> for SizeTensor {
    type Error = Error;

    fn try_from(from: SizeTensorUnchecked) -> Result<Self, Self::Error> {
        let SizeTensorUnchecked { h, w } = from;
        match (h.size2()?, w.size2()?) {
            ((h_len, 1), (w_len, 1)) => ensure!(h_len == w_len, "size mismatch"),
            _ => bail!("size mismatch"),
        };
        ensure!(
            hashset! {
                h.device(),
                w.device(),
            }
            .len()
                == 1,
            "device mismatch"
        );
        Ok(Self { h, w })
    }
}

impl From<SizeTensor> for SizeTensorUnchecked {
    fn from(from: SizeTensor) -> Self {
        let SizeTensor { h, w } = from;
        Self { h, w }
    }
}
