use super::cycxhw::{CyCxHWTensor, CyCxHWTensorUnchecked};
use crate::common::*;

#[derive(Debug, TensorLike, Getters)]
pub struct DetectionTensor {
    /// The box parameters in CyCxHW format.
    #[get = "pub"]
    pub(crate) cycxhw: CyCxHWTensor,
    /// The objectness scores in shape `[batch, 1]`.
    #[get = "pub"]
    pub(crate) obj: Tensor,
    /// The classification scores in shape `[batch, class]`.
    #[get = "pub"]
    pub(crate) class: Tensor,
}

#[derive(Debug, TensorLike)]
pub struct DetectionTensorUnchecked {
    pub cycxhw: CyCxHWTensorUnchecked,
    pub obj: Tensor,
    pub class: Tensor,
}

impl DetectionTensor {
    pub fn num_samples(&self) -> i64 {
        self.class.size2().unwrap().0
    }

    pub fn num_classes(&self) -> i64 {
        self.class.size2().unwrap().1
    }

    pub fn cat<T>(iter: impl IntoIterator<Item = T>) -> Self
    where
        T: Borrow<Self>,
    {
        let (cycxhw_vec, obj_vec, class_vec) = iter
            .into_iter()
            .map(|detection| {
                let Self { cycxhw, obj, class } = detection.borrow().shallow_clone();
                (cycxhw, obj, class)
            })
            .unzip_n_vec();

        Self {
            cycxhw: CyCxHWTensor::cat(cycxhw_vec),
            obj: Tensor::cat(&obj_vec, 0),
            class: Tensor::cat(&class_vec, 0),
        }
    }
}

impl TryFrom<DetectionTensorUnchecked> for DetectionTensor {
    type Error = Error;

    fn try_from(from: DetectionTensorUnchecked) -> Result<Self, Self::Error> {
        let DetectionTensorUnchecked { cycxhw, obj, class } = from;
        let cycxhw: CyCxHWTensor = cycxhw.try_into()?;

        match (obj.size2()?, class.size2()?) {
            ((obj_len, 1), (class_len, _num_class)) => {
                ensure!(
                    cycxhw.num_samples() == obj_len && obj_len == class_len,
                    "size mismatch"
                );
            }
            _ => bail!("size mismatch"),
        }

        Ok(Self { cycxhw, obj, class })
    }
}

impl From<DetectionTensor> for DetectionTensorUnchecked {
    fn from(from: DetectionTensor) -> Self {
        let DetectionTensor { cycxhw, obj, class } = from;
        Self {
            cycxhw: cycxhw.into(),
            obj,
            class,
        }
    }
}
