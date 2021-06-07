use crate::{
    common::*,
    compound_tensor::{CyCxHWTensor, CyCxHWTensorUnchecked},
};

#[derive(Debug, TensorLike, Getters)]
pub struct DenseDetectionSamples {
    /// The box parameters in CyCxHW format.
    #[get = "pub"]
    pub(crate) cycxhw: CyCxHWTensor,
    /// The objectness scores in shape `[batch, 1]`.
    #[get = "pub"]
    pub(crate) obj_logit: Tensor,
    /// The classification scores in shape `[batch, class]`.
    #[get = "pub"]
    pub(crate) class_logit: Tensor,
}

#[derive(Debug, TensorLike)]
pub struct DenseDetectionSamplesUnchecked {
    pub cycxhw: CyCxHWTensorUnchecked,
    pub obj_logit: Tensor,
    pub class_logit: Tensor,
}

impl DenseDetectionSamples {
    pub fn num_samples(&self) -> i64 {
        self.class_logit.size2().unwrap().0
    }

    pub fn num_classes(&self) -> i64 {
        self.class_logit.size2().unwrap().1
    }

    /// Compute confidence, objectness score times classification score.
    pub fn confidence(&self) -> Tensor {
        self.obj_prob() * self.class_prob()
    }

    pub fn obj_prob(&self) -> Tensor {
        self.obj_logit.sigmoid()
    }

    pub fn class_prob(&self) -> Tensor {
        self.class_logit.sigmoid()
    }

    pub fn cat<T>(iter: impl IntoIterator<Item = T>) -> Self
    where
        T: Borrow<Self>,
    {
        let (cycxhw_vec, obj_vec, class_vec) = iter
            .into_iter()
            .map(|detection| {
                let Self {
                    cycxhw,
                    obj_logit,
                    class_logit,
                } = detection.borrow().shallow_clone();
                (cycxhw, obj_logit, class_logit)
            })
            .unzip_n_vec();

        Self {
            cycxhw: CyCxHWTensor::cat(cycxhw_vec),
            obj_logit: Tensor::cat(&obj_vec, 0),
            class_logit: Tensor::cat(&class_vec, 0),
        }
    }
}

impl TryFrom<DenseDetectionSamplesUnchecked> for DenseDetectionSamples {
    type Error = Error;

    fn try_from(from: DenseDetectionSamplesUnchecked) -> Result<Self, Self::Error> {
        let DenseDetectionSamplesUnchecked {
            cycxhw,
            obj_logit,
            class_logit,
        } = from;
        let cycxhw: CyCxHWTensor = cycxhw.try_into()?;

        match (obj_logit.size2()?, class_logit.size2()?) {
            ((obj_len, 1), (class_len, _num_class)) => {
                ensure!(
                    cycxhw.num_samples() == obj_len && obj_len == class_len,
                    "size mismatch"
                );
            }
            _ => bail!("size mismatch"),
        }

        Ok(Self {
            cycxhw,
            obj_logit,
            class_logit,
        })
    }
}

impl From<DenseDetectionSamples> for DenseDetectionSamplesUnchecked {
    fn from(from: DenseDetectionSamples) -> Self {
        let DenseDetectionSamples {
            cycxhw,
            obj_logit,
            class_logit,
        } = from;
        Self {
            cycxhw: cycxhw.into(),
            obj_logit,
            class_logit,
        }
    }
}
