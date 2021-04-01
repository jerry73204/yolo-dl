use super::cycxhw::{CyCxHWTensor, CyCxHWTensorUnchecked};
use crate::{common::*, detection::Label, unit::Unit};
use num_traits::NumCast;

#[derive(Debug, TensorLike, Getters)]
pub struct LabelTensor {
    #[get = "pub"]
    pub(crate) cycxhw: CyCxHWTensor,
    #[get = "pub"]
    pub(crate) class: Tensor,
}

#[derive(Debug, TensorLike)]
pub struct LabelTensorUnchecked {
    pub cycxhw: CyCxHWTensorUnchecked,
    pub class: Tensor,
}

impl LabelTensor {
    pub fn cat<T>(iter: impl IntoIterator<Item = T>) -> Self
    where
        T: Borrow<Self>,
    {
        let (cycxhw_vec, class_vec) = iter
            .into_iter()
            .map(|label| {
                let Self { cycxhw, class } = label.borrow().shallow_clone();
                (cycxhw, class)
            })
            .unzip_n_vec();

        Self {
            cycxhw: CyCxHWTensor::cat(cycxhw_vec),
            class: Tensor::cat(&class_vec, 0),
        }
    }
}

impl TryFrom<LabelTensorUnchecked> for LabelTensor {
    type Error = Error;

    fn try_from(from: LabelTensorUnchecked) -> Result<Self, Self::Error> {
        let LabelTensorUnchecked { cycxhw, class } = from;
        let cycxhw: CyCxHWTensor = cycxhw.try_into()?;

        match class.size2()? {
            (class_len, 1) => {
                ensure!(cycxhw.num_samples() == class_len, "size mismatch");
            }
            _ => bail!("size mismatch"),
        }

        Ok(Self { cycxhw, class })
    }
}

impl From<LabelTensor> for LabelTensorUnchecked {
    fn from(from: LabelTensor) -> Self {
        let LabelTensor { cycxhw, class } = from;
        Self {
            cycxhw: cycxhw.into(),
            class,
        }
    }
}

impl<T, U> FromIterator<Label<T, U>> for LabelTensor
where
    T: Float,
    U: Unit,
{
    fn from_iter<I: IntoIterator<Item = Label<T, U>>>(iter: I) -> Self {
        let (cy, cx, h, w, class) = iter
            .into_iter()
            .map(|label| {
                let [cy, cx, h, w] = label.cycxhw.cycxhw_params();
                let class = label.class;
                (
                    <f32 as NumCast>::from(cy).unwrap(),
                    <f32 as NumCast>::from(cx).unwrap(),
                    <f32 as NumCast>::from(h).unwrap(),
                    <f32 as NumCast>::from(w).unwrap(),
                    class as i64,
                )
            })
            .unzip_n_vec();

        LabelTensorUnchecked {
            cycxhw: CyCxHWTensorUnchecked {
                cy: Tensor::of_slice(&cy),
                cx: Tensor::of_slice(&cx),
                h: Tensor::of_slice(&h),
                w: Tensor::of_slice(&w),
            },
            class: Tensor::of_slice(&class),
        }
        .try_into()
        .unwrap()
    }
}

impl<'a, T, U> FromIterator<&'a Label<T, U>> for LabelTensor
where
    T: Float,
    U: Unit,
{
    fn from_iter<I: IntoIterator<Item = &'a Label<T, U>>>(iter: I) -> Self {
        let (cy_vec, cx_vec, h_vec, w_vec, class_vec) = iter
            .into_iter()
            .map(|label| {
                let [cy, cx, h, w] = label.cycxhw.cycxhw_params();
                let class = label.class;
                (
                    <f32 as NumCast>::from(cy).unwrap(),
                    <f32 as NumCast>::from(cx).unwrap(),
                    <f32 as NumCast>::from(h).unwrap(),
                    <f32 as NumCast>::from(w).unwrap(),
                    class as i64,
                )
            })
            .unzip_n_vec();

        LabelTensorUnchecked {
            cycxhw: CyCxHWTensorUnchecked {
                cy: Tensor::of_slice(&cy_vec).view([-1, 1]),
                cx: Tensor::of_slice(&cx_vec).view([-1, 1]),
                h: Tensor::of_slice(&h_vec).view([-1, 1]),
                w: Tensor::of_slice(&w_vec).view([-1, 1]),
            },
            class: Tensor::of_slice(&class_vec).view([-1, 1]),
        }
        .try_into()
        .unwrap()
    }
}
