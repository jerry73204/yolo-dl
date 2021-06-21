use super::{area::AreaTensor, cycxhw::CyCxHWTensor, size::SizeTensor};
use crate::{
    bbox::{CyCxHW, Rect},
    common::*,
    unit::Unit,
    utils::IntoTchElement,
};

/// Checked tensor of batched box parameters in TLBR format.
#[derive(Debug, TensorLike, Getters)]
pub struct TLBRTensor {
    /// The top parameter in shape `[batch, 1]`.
    #[get = "pub"]
    pub(crate) t: Tensor,
    /// The left parameter in shape `[batch, 1]`.
    #[get = "pub"]
    pub(crate) l: Tensor,
    /// The bottom parameter in shape `[batch, 1]`.
    #[get = "pub"]
    pub(crate) b: Tensor,
    /// The right parameter in shape `[batch, 1]`.
    #[get = "pub"]
    pub(crate) r: Tensor,
}

/// Unchecked tensor of batched box parameters in TLBR format.
#[derive(Debug, TensorLike)]
pub struct TLBRTensorUnchecked {
    /// The top parameter in shape `[batch, 1]`.
    pub t: Tensor,
    /// The left parameter in shape `[batch, 1]`.
    pub l: Tensor,
    /// The bottom parameter in shape `[batch, 1]`.
    pub b: Tensor,
    /// The right parameter in shape `[batch, 1]`.
    pub r: Tensor,
}

impl TLBRTensor {
    pub fn num_samples(&self) -> i64 {
        let (num, _) = self.t.size2().unwrap();
        num
    }

    pub fn device(&self) -> Device {
        self.t.device()
    }

    pub fn select(&self, index: i64) -> Self {
        let Self { t, l, b, r } = self;
        let range = index..(index + 1);
        Self {
            t: t.i((range.clone(), ..)),
            l: l.i((range.clone(), ..)),
            b: b.i((range.clone(), ..)),
            r: r.i((range, ..)),
        }
    }

    pub fn index_select(&self, indexes: &Tensor) -> Self {
        let Self { t, l, b, r } = self;
        let t = t.index_select(0, indexes);
        let l = l.index_select(0, indexes);
        let b = b.index_select(0, indexes);
        let r = r.index_select(0, indexes);
        Self { t, l, b, r }
    }

    /// Compute the box size.
    pub fn size(&self) -> SizeTensor {
        let Self { t, l, b, r } = self;
        let h = b - t;
        let w = r - l;

        SizeTensor { h, w }
    }

    /// Compute the box area.
    pub fn area(&self) -> AreaTensor {
        let SizeTensor { h, w } = self.size();
        let area = h * w;
        AreaTensor { area }
    }

    /// Compute the intersection area with the other box tensor.
    pub fn intersect_area_with(&self, other: &Self) -> AreaTensor {
        let Self {
            t: lhs_t,
            l: lhs_l,
            b: lhs_b,
            r: lhs_r,
        } = self;
        let Self {
            t: rhs_t,
            l: rhs_l,
            b: rhs_b,
            r: rhs_r,
        } = other;

        let max_t = lhs_t.max1(rhs_t);
        let max_l = lhs_l.max1(rhs_l);
        let min_b = lhs_b.min1(rhs_b);
        let min_r = lhs_r.min1(rhs_r);

        let inner_h = (min_b - max_t).clamp_min(0.0);
        let inner_w = (min_r - max_l).clamp_min(0.0);

        let area = inner_h * inner_w;

        AreaTensor { area }
    }

    /// Compute the rectangle closure with the other box tensor.
    pub fn closure_with(&self, other: &Self) -> Self {
        let Self {
            t: lhs_t,
            l: lhs_l,
            b: lhs_b,
            r: lhs_r,
        } = self;
        let Self {
            t: rhs_t,
            l: rhs_l,
            b: rhs_b,
            r: rhs_r,
        } = other;

        let min_t = lhs_t.min1(rhs_t);
        let min_l = lhs_l.min1(rhs_l);
        let max_b = lhs_b.max1(rhs_b);
        let max_r = lhs_r.max1(rhs_r);

        Self {
            t: min_t,
            l: min_l,
            b: max_b,
            r: max_r,
        }
    }

    /// Compute the Hausdorff distance with the other box tensor.
    pub fn hausdorff_distance_to(&self, other: &Self) -> Tensor {
        let TLBRTensor {
            t: tl,
            l: ll,
            b: bl,
            r: rl,
        } = self;
        let TLBRTensor {
            t: tr,
            l: lr,
            b: br,
            r: rr,
        } = other;

        let dt = tr - tl;
        let dl = lr - ll;
        let db = bl - br;
        let dr = rl - rr;

        let dt_l = dt.clamp_min(0.0);
        let dl_l = dl.clamp_min(0.0);
        let db_l = db.clamp_min(0.0);
        let dr_l = dr.clamp_min(0.0);

        let dt_r = (-&dt).clamp_min(0.0);
        let dl_r = (-&dl).clamp_min(0.0);
        let db_r = (-&db).clamp_min(0.0);
        let dr_r = (-&dr).clamp_min(0.0);

        (dt_l.pow(2.0) + dl_l.pow(2.0))
            .max1(&(dt_l.pow(2.0) + dr_l.pow(2.0)))
            .max1(&(db_l.pow(2.0) + dl_l.pow(2.0)))
            .max1(&(db_l.pow(2.0) + dr_l.pow(2.0)))
            .max1(&(dt_r.pow(2.0) + dl_r.pow(2.0)))
            .max1(&(dt_r.pow(2.0) + dr_r.pow(2.0)))
            .max1(&(db_r.pow(2.0) + dl_r.pow(2.0)))
            .max1(&(db_r.pow(2.0) + dr_r.pow(2.0)))
            .sqrt()
    }
}

impl TryFrom<TLBRTensorUnchecked> for TLBRTensor {
    type Error = Error;

    fn try_from(from: TLBRTensorUnchecked) -> Result<Self, Self::Error> {
        let TLBRTensorUnchecked { t, l, b, r } = from;
        match (t.size2()?, l.size2()?, b.size2()?, r.size2()?) {
            ((t_len, 1), (l_len, 1), (b_len, 1), (r_len, 1)) => ensure!(
                t_len == l_len && t_len == b_len && t_len == r_len,
                "size mismatch"
            ),
            _ => bail!("size mismatch"),
        };
        ensure!(
            hashset! {
                t.device(),
                l.device(),
                b.device(),
                r.device(),
            }
            .len()
                == 1,
            "device mismatch"
        );
        Ok(Self { t, l, b, r })
    }
}

impl From<TLBRTensor> for TLBRTensorUnchecked {
    fn from(from: TLBRTensor) -> Self {
        let TLBRTensor { t, l, b, r } = from;
        Self { t, l, b, r }
    }
}

impl From<&CyCxHWTensor> for TLBRTensor {
    fn from(from: &CyCxHWTensor) -> Self {
        let CyCxHWTensor { cy, cx, h, w } = from;

        let t = cy - h / 2.0;
        let b = cy + h / 2.0;
        let l = cx - w / 2.0;
        let r = cx + w / 2.0;

        Self { t, l, b, r }
    }
}

impl<T, U> From<&[&CyCxHW<T, U>]> for TLBRTensor
where
    T: Float + IntoTchElement,
    U: Unit,
{
    fn from(from: &[&CyCxHW<T, U>]) -> Self {
        bboxes_to_tlbr_tensor(from)
    }
}

impl<T, U> From<&[CyCxHW<T, U>]> for TLBRTensor
where
    T: Float + IntoTchElement,
    U: Unit,
{
    fn from(from: &[CyCxHW<T, U>]) -> Self {
        bboxes_to_tlbr_tensor(from)
    }
}

impl<T, U> From<&Vec<CyCxHW<T, U>>> for TLBRTensor
where
    T: Float + IntoTchElement,
    U: Unit,
{
    fn from(from: &Vec<CyCxHW<T, U>>) -> Self {
        bboxes_to_tlbr_tensor(from.as_slice())
    }
}

impl<T, U> From<Vec<CyCxHW<T, U>>> for TLBRTensor
where
    T: Float + IntoTchElement,
    U: Unit,
{
    fn from(from: Vec<CyCxHW<T, U>>) -> Self {
        bboxes_to_tlbr_tensor(from.as_slice())
    }
}

impl<T, U> TryFrom<&TLBRTensor> for Vec<CyCxHW<T, U>>
where
    T: Float,
    U: Unit,
{
    type Error = Error;

    fn try_from(from: &TLBRTensor) -> Result<Self, Self::Error> {
        (&CyCxHWTensor::from(from)).try_into()
    }
}

impl<T, U> TryFrom<TLBRTensor> for Vec<CyCxHW<T, U>>
where
    T: Float,
    U: Unit,
{
    type Error = Error;

    fn try_from(from: TLBRTensor) -> Result<Self, Self::Error> {
        (&from).try_into()
    }
}

fn bboxes_to_tlbr_tensor<B, T, U>(bboxes: &[B]) -> TLBRTensor
where
    B: AsRef<CyCxHW<T, U>>,
    T: Float + IntoTchElement,
    U: Unit,
{
    let (t_vec, l_vec, b_vec, r_vec) = bboxes
        .iter()
        .map(|bbox| {
            let tlbr = bbox.as_ref().to_tlbr();
            (
                tlbr.t().into_tch_element(),
                tlbr.l().into_tch_element(),
                tlbr.b().into_tch_element(),
                tlbr.r().into_tch_element(),
            )
        })
        .unzip_n_vec();

    let t_tensor = Tensor::of_slice(&t_vec);
    let l_tensor = Tensor::of_slice(&l_vec);
    let b_tensor = Tensor::of_slice(&b_vec);
    let r_tensor = Tensor::of_slice(&r_vec);

    TLBRTensor {
        t: t_tensor,
        l: l_tensor,
        b: b_tensor,
        r: r_tensor,
    }
}
