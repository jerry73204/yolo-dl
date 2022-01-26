use crate::common::*;
use getset::Getters;
use tch_goodies::{CyCxHWTensor, CyCxHWTensorUnchecked};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatchGrid {
    Rect2,
    Rect4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BoxMetric {
    IoU,
    GIoU,
    DIoU,
    CIoU,
    Hausdorff,
}

#[derive(Debug, TensorLike)]
struct Grid {
    pub cycxhw: Tensor,
    pub objectness: Tensor,
    pub classification: Tensor,
}

#[derive(Debug, TensorLike)]
pub struct PredInstancesUnchecked {
    pub cycxhw: CyCxHWTensorUnchecked,
    pub objectness: Tensor,
    pub dense_class: Tensor,
}

#[derive(Debug, TensorLike)]
pub struct TargetInstancesUnchecked {
    pub cycxhw: CyCxHWTensorUnchecked,
    pub sparse_class: Tensor,
}

#[derive(Debug, TensorLike, Getters)]
pub struct PredInstances {
    #[get = "pub"]
    cycxhw: CyCxHWTensor,
    #[get = "pub"]
    objectness: Tensor,
    #[get = "pub"]
    dense_class: Tensor,
}

#[derive(Debug, TensorLike, Getters)]
pub struct TargetInstances {
    #[get = "pub"]
    cycxhw: CyCxHWTensor,
    #[get = "pub"]
    sparse_class: Tensor,
}

impl TryFrom<PredInstancesUnchecked> for PredInstances {
    type Error = Error;

    fn try_from(from: PredInstancesUnchecked) -> Result<Self, Self::Error> {
        let PredInstancesUnchecked {
            cycxhw,
            objectness,
            dense_class,
        } = from;

        let cycxhw: CyCxHWTensor = cycxhw.try_into()?;
        let cycxhw_len = cycxhw.num_samples();
        let (obj_len, obj_entries) = objectness.size2()?;
        let (class_len, _classes) = dense_class.size2()?;
        ensure!(
            obj_entries == 1 && cycxhw_len == obj_len && cycxhw_len == class_len,
            "size mismatch"
        );

        Ok(Self {
            cycxhw,
            objectness,
            dense_class,
        })
    }
}

impl TryFrom<TargetInstancesUnchecked> for TargetInstances {
    type Error = Error;

    fn try_from(from: TargetInstancesUnchecked) -> Result<Self, Self::Error> {
        let TargetInstancesUnchecked {
            cycxhw,
            sparse_class,
        } = from;

        let cycxhw: CyCxHWTensor = cycxhw.try_into()?;
        let cycxhw_len = cycxhw.num_samples();
        let (class_len, _classes) = sparse_class.size2()?;
        ensure!(cycxhw_len == class_len, "size mismatch");

        Ok(Self {
            cycxhw,
            sparse_class,
        })
    }
}

impl From<PredInstances> for PredInstancesUnchecked {
    fn from(from: PredInstances) -> Self {
        let PredInstances {
            cycxhw,
            objectness,
            dense_class,
        } = from;
        Self {
            cycxhw: cycxhw.into(),
            objectness,
            dense_class,
        }
    }
}

impl From<TargetInstances> for TargetInstancesUnchecked {
    fn from(from: TargetInstances) -> Self {
        let TargetInstances {
            cycxhw,
            sparse_class,
        } = from;
        Self {
            cycxhw: cycxhw.into(),
            sparse_class,
        }
    }
}
