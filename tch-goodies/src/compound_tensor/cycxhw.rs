use super::{area::AreaTensor, size::SizeTensor, tlbr::TLBRTensor};
use crate::{bbox::CyCxHW, common::*, unit::Unit, utils::EPSILON};
use num_traits::NumCast;

/// Checked tensor of batched box parameters in CyCxHW format.
#[derive(Debug, TensorLike, Getters)]
pub struct CyCxHWTensor {
    /// The center y parameter in shape `[batch, 1]`.
    #[get = "pub"]
    pub(crate) cy: Tensor,
    /// The center x parameter in shape `[batch, 1]`.
    #[get = "pub"]
    pub(crate) cx: Tensor,
    /// The height parameter in shape `[batch, 1]`.
    #[get = "pub"]
    pub(crate) h: Tensor,
    /// The width parameter in shape `[batch, 1]`.
    #[get = "pub"]
    pub(crate) w: Tensor,
}

/// Unchecked tensor of batched box parameters in CyCxHW format.
#[derive(Debug, TensorLike)]
pub struct CyCxHWTensorUnchecked {
    /// The center y parameter in shape `[batch, 1]`.
    pub cy: Tensor,
    /// The center x parameter in shape `[batch, 1]`.
    pub cx: Tensor,
    /// The height parameter in shape `[batch, 1]`.
    pub h: Tensor,
    /// The width parameter in shape `[batch, 1]`.
    pub w: Tensor,
}

impl CyCxHWTensor {
    pub fn num_samples(&self) -> i64 {
        let (num, _) = self.cy.size2().unwrap();
        num
    }

    /// Compute box area.
    pub fn area(&self) -> AreaTensor {
        let Self { h, w, .. } = self;
        let area = h * w;
        AreaTensor { area }
    }

    /// Compute box size.
    pub fn size(&self) -> SizeTensor {
        let Self { h, w, .. } = self;
        SizeTensor {
            h: h.shallow_clone(),
            w: w.shallow_clone(),
        }
    }

    /// Compute the intersection area with the other box tensor.
    pub fn intersect_area_with(&self, other: &Self) -> AreaTensor {
        TLBRTensor::from(self).intersect_area_with(&TLBRTensor::from(other))
    }

    /// Compute the rectanble closure with the other box tensor.
    pub fn closure_with(&self, other: &Self) -> CyCxHWTensor {
        (&TLBRTensor::from(self).closure_with(&TLBRTensor::from(other))).into()
    }

    /// Compute the IoU score with the other box tensor.
    pub fn iou_with(&self, other: &Self) -> Tensor {
        let inter_area = self.intersect_area_with(other);
        let outer_area = self.area().area() + other.area().area() - inter_area.area() + EPSILON;
        let iou = inter_area.area() / outer_area;
        iou
    }

    /// Compute the GIoU score with the other box tensor.
    pub fn giou_with(&self, other: &Self) -> Tensor {
        let inter_area = self.intersect_area_with(other);
        let outer_area = self.area().area() + other.area().area() - inter_area.area() + EPSILON;
        let closure = self.closure_with(&other);
        let closure_area = closure.area();
        let iou = inter_area.area() / &outer_area;
        iou - (closure_area.area() - &outer_area) / (closure_area.area() + EPSILON)
    }

    /// Compute the DIoU score with the other box tensor.
    pub fn diou_with(&self, other: &Self) -> Tensor {
        let iou = self.iou_with(other);

        let closure = TLBRTensor::from(self).closure_with(&TLBRTensor::from(other));
        let closure_size = closure.size();

        let diagonal_square = closure_size.h().pow(2.0) + closure_size.w().pow(2.0) + EPSILON;
        let center_dist_square =
            (self.cy() - other.cy()).pow(2.0) + (self.cx() - other.cx()).pow(2.0);

        iou - center_dist_square / diagonal_square
    }

    /// Compute the CIoU score with the other box tensor.
    pub fn ciou_with(&self, other: &Self) -> Tensor {
        use std::f64::consts::PI;

        let iou = self.iou_with(other);

        let closure = TLBRTensor::from(self).closure_with(&TLBRTensor::from(other));
        let closure_size = closure.size();

        let pred_angle = self.h().atan2(&self.w());
        let target_angle = other.h().atan2(&other.w());

        let diagonal_square = closure_size.h().pow(2.0) + closure_size.w().pow(2.0) + EPSILON;
        let center_dist_square =
            (self.cy() - other.cy()).pow(2.0) + (self.cx() - other.cx()).pow(2.0);

        let shape_loss = (&pred_angle - &target_angle).pow(2.0) * 4.0 / PI.powi(2);
        let shape_loss_coef = tch::no_grad(|| &shape_loss / (1.0 - &iou + &shape_loss));

        iou - center_dist_square / diagonal_square + shape_loss_coef * shape_loss
    }

    /// Compute the Hausdorff distance with the other box tensor.
    pub fn hausdorff_distance_to(&self, other: &Self) -> Tensor {
        TLBRTensor::from(self).hausdorff_distance_to(&TLBRTensor::from(other))
    }

    pub fn cat<T>(iter: impl IntoIterator<Item = T>) -> Self
    where
        T: Borrow<Self>,
    {
        let (cy_vec, cx_vec, h_vec, w_vec) = iter
            .into_iter()
            .map(|cycxhw| {
                let Self { cy, cx, h, w } = cycxhw.borrow().shallow_clone();
                (cy, cx, h, w)
            })
            .unzip_n_vec();

        Self {
            cy: Tensor::cat(&cy_vec, 0),
            cx: Tensor::cat(&cx_vec, 0),
            h: Tensor::cat(&h_vec, 0),
            w: Tensor::cat(&w_vec, 0),
        }
    }
}

impl TryFrom<CyCxHWTensorUnchecked> for CyCxHWTensor {
    type Error = Error;

    fn try_from(from: CyCxHWTensorUnchecked) -> Result<Self, Self::Error> {
        let CyCxHWTensorUnchecked { cy, cx, h, w } = from;
        match (cy.size2()?, cx.size2()?, h.size2()?, w.size2()?) {
            ((cy_len, 1), (cx_len, 1), (h_len, 1), (w_len, 1)) => ensure!(
                cy_len == cx_len && cy_len == h_len && cy_len == w_len,
                "size mismatch"
            ),
            _ => bail!("size mismatch"),
        };
        ensure!(
            hashset! {
                cy.device(),
                cx.device(),
                h.device(),
                w.device(),
            }
            .len()
                == 1,
            "device mismatch"
        );
        Ok(Self { cy, cx, h, w })
    }
}

impl From<CyCxHWTensor> for CyCxHWTensorUnchecked {
    fn from(from: CyCxHWTensor) -> Self {
        let CyCxHWTensor { cy, cx, h, w } = from;
        Self { cy, cx, h, w }
    }
}

impl From<&TLBRTensor> for CyCxHWTensor {
    fn from(from: &TLBRTensor) -> Self {
        let TLBRTensor { t, l, b, r } = from;
        let h = b - t;
        let w = r - l;
        let cy = t + &h / 2.0;
        let cx = l + &w / 2.0;
        Self { cy, cx, h, w }
    }
}

impl<T, U> TryFrom<&CyCxHWTensor> for Vec<CyCxHW<T, U>>
where
    T: Float,
    U: Unit,
{
    type Error = Error;

    fn try_from(from: &CyCxHWTensor) -> Result<Self, Self::Error> {
        let bboxes: Option<Vec<_>> = izip!(
            Vec::<f32>::from(from.cy()),
            Vec::<f32>::from(from.cx()),
            Vec::<f32>::from(from.h()),
            Vec::<f32>::from(from.w()),
        )
        .map(|(cy, cx, h, w)| {
            let cycxhw =
                CyCxHW::<T, U>::from_cycxhw(T::from(cy)?, T::from(cx)?, T::from(h)?, T::from(w)?)
                    .unwrap();
            Some(cycxhw)
        })
        .collect();

        bboxes.ok_or_else(|| format_err!("casting error"))
    }
}

impl<T, U> FromIterator<CyCxHW<T, U>> for CyCxHWTensor
where
    T: Float,
    U: Unit,
{
    fn from_iter<I: IntoIterator<Item = CyCxHW<T, U>>>(iter: I) -> Self {
        let (cy, cx, h, w) = iter
            .into_iter()
            .map(|cycxhw| {
                let [cy, cx, h, w] = cycxhw.cycxhw_params();
                (
                    <f32 as NumCast>::from(cy).unwrap(),
                    <f32 as NumCast>::from(cx).unwrap(),
                    <f32 as NumCast>::from(h).unwrap(),
                    <f32 as NumCast>::from(w).unwrap(),
                )
            })
            .unzip_n_vec();

        CyCxHWTensorUnchecked {
            cy: Tensor::of_slice(&cy),
            cx: Tensor::of_slice(&cx),
            h: Tensor::of_slice(&h),
            w: Tensor::of_slice(&w),
        }
        .try_into()
        .unwrap()
    }
}

impl<'a, T, U> FromIterator<&'a CyCxHW<T, U>> for CyCxHWTensor
where
    T: Float,
    U: Unit,
{
    fn from_iter<I: IntoIterator<Item = &'a CyCxHW<T, U>>>(iter: I) -> Self {
        let (cy, cx, h, w) = iter
            .into_iter()
            .map(|cycxhw| {
                let [cy, cx, h, w] = cycxhw.cycxhw_params();
                (
                    <f32 as NumCast>::from(cy).unwrap(),
                    <f32 as NumCast>::from(cx).unwrap(),
                    <f32 as NumCast>::from(h).unwrap(),
                    <f32 as NumCast>::from(w).unwrap(),
                )
            })
            .unzip_n_vec();

        CyCxHWTensorUnchecked {
            cy: Tensor::of_slice(&cy),
            cx: Tensor::of_slice(&cx),
            h: Tensor::of_slice(&h),
            w: Tensor::of_slice(&w),
        }
        .try_into()
        .unwrap()
    }
}
