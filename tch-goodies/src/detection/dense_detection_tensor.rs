use crate::{common::*, size::GridSize};

/// Represents the output feature map of a layer.
///
/// Every belonging tensor has shape `[batch, entry, anchor, height, width]`.
#[derive(Debug, TensorLike, Getters)]
pub struct DenseDetectionTensor {
    pub(super) inner: DenseDetectionTensorUnchecked,
}

impl DenseDetectionTensor {
    pub fn batch_size(&self) -> usize {
        let (batch_size, _, _, _, _) = self.cy.size5().unwrap();
        batch_size as usize
    }

    pub fn num_classes(&self) -> usize {
        let (_, num_classes, _, _, _) = self.class_logit.size5().unwrap();
        num_classes as usize
    }

    pub fn num_anchors(&self) -> usize {
        let (_, _, num_anchors, _, _) = self.cy.size5().unwrap();
        num_anchors as usize
    }

    pub fn height(&self) -> usize {
        let (_, _, _, height, _) = self.cy.size5().unwrap();
        height as usize
    }

    pub fn width(&self) -> usize {
        let (_, _, _, _, width) = self.cy.size5().unwrap();
        width as usize
    }

    pub fn obj_prob(&self) -> Tensor {
        self.inner.obj_logit.sigmoid()
    }

    pub fn class_prob(&self) -> Tensor {
        self.inner.class_logit.sigmoid()
    }

    /// Compute confidence, objectness score times classification score.
    pub fn confidence(&self) -> Tensor {
        self.obj_prob() * self.class_prob()
    }

    pub fn index_select_batch(&self, index: &Tensor) -> Result<Self> {
        ensure!(index.dim() == 1 && index.kind() == Kind::Int64);

        let batch_size = self.batch_size();
        let ok: bool = index.lt(batch_size as i64).all().into();
        ensure!(ok, "batch index exceeds batch size {}", batch_size);

        let DenseDetectionTensorUnchecked {
            cy,
            cx,
            h,
            w,
            obj_logit,
            class_logit,
            anchors,
        } = &self.inner;

        let cy = cy.index_select(0, index);
        let cx = cx.index_select(0, index);
        let h = h.index_select(0, index);
        let w = w.index_select(0, index);
        let obj_logit = obj_logit.index_select(0, index);
        let class_logit = class_logit.index_select(0, index);

        Ok(Self {
            inner: DenseDetectionTensorUnchecked {
                cy,
                cx,
                h,
                w,
                obj_logit,
                class_logit,
                anchors: anchors.to_owned(),
            },
        })
    }

    pub fn cat_batch(tensors: impl IntoIterator<Item = impl Borrow<Self>>) -> Result<Self> {
        let (
            batch_size_set,
            num_classes_set,
            feature_h_set,
            feature_w_set,
            anchors_set,
            cy_vec,
            cx_vec,
            h_vec,
            w_vec,
            obj_vec,
            class_vec,
        ): (
            HashSet<_>,
            HashSet<_>,
            HashSet<_>,
            HashSet<_>,
            HashSet<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
        ) = tensors
            .into_iter()
            .map(|tensor| {
                let tensor = tensor.borrow().shallow_clone();
                let batch_size = tensor.batch_size();
                let num_classes = tensor.num_classes();
                let feature_h = tensor.height();
                let feature_w = tensor.width();

                let DenseDetectionTensorUnchecked {
                    cy,
                    cx,
                    h,
                    w,
                    obj_logit,
                    class_logit,
                    anchors,
                } = tensor.into();
                (
                    batch_size,
                    num_classes,
                    feature_h,
                    feature_w,
                    anchors,
                    cy,
                    cx,
                    h,
                    w,
                    obj_logit,
                    class_logit,
                )
            })
            .unzip_n();

        ensure!(batch_size_set.len() == 1);
        ensure!(num_classes_set.len() == 1);
        ensure!(anchors_set.len() == 1);
        ensure!(feature_h_set.len() == 1);
        ensure!(feature_w_set.len() == 1);

        let cy = Tensor::cat(&cy_vec, 0);
        let cx = Tensor::cat(&cx_vec, 0);
        let h = Tensor::cat(&h_vec, 0);
        let w = Tensor::cat(&w_vec, 0);
        let obj_logit = Tensor::cat(&obj_vec, 0);
        let class_logit = Tensor::cat(&class_vec, 0);
        let anchors = anchors_set.into_iter().next().unwrap();

        Ok(Self {
            inner: DenseDetectionTensorUnchecked {
                cy,
                cx,
                h,
                w,
                obj_logit,
                class_logit,
                anchors,
            },
        })
    }

    pub fn cat_height(tensors: impl IntoIterator<Item = impl Borrow<Self>>) -> Result<Self> {
        todo!();
    }

    pub fn cat_width(tensors: impl IntoIterator<Item = impl Borrow<Self>>) -> Result<Self> {
        todo!();
    }
}

#[derive(Debug, TensorLike)]
pub struct DenseDetectionTensorUnchecked {
    /// The bounding box center y position in ratio unit. It has 1 entry.
    pub cy: Tensor,
    /// The bounding box center x position in ratio unit. It has 1 entry.
    pub cx: Tensor,
    /// The bounding box height in ratio unit. It has 1 entry.
    pub h: Tensor,
    /// The bounding box width in ratio unit. It has 1 entry.
    pub w: Tensor,
    /// The likelihood score an object in the position. It has 1 entry.
    pub obj_logit: Tensor,
    /// The scores the object is of that class. It number of entries is the number of classes.
    pub class_logit: Tensor,
    #[tensor_like(clone)]
    pub anchors: Vec<GridSize<R64>>,
}

impl Deref for DenseDetectionTensor {
    type Target = DenseDetectionTensorUnchecked;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Borrow<DenseDetectionTensorUnchecked> for DenseDetectionTensor {
    fn borrow(&self) -> &DenseDetectionTensorUnchecked {
        &self.inner
    }
}

impl From<DenseDetectionTensor> for DenseDetectionTensorUnchecked {
    fn from(from: DenseDetectionTensor) -> Self {
        from.inner
    }
}

impl TryFrom<DenseDetectionTensorUnchecked> for DenseDetectionTensor {
    type Error = Error;

    fn try_from(from: DenseDetectionTensorUnchecked) -> Result<Self, Self::Error> {
        let DenseDetectionTensorUnchecked {
            cy,
            cx,
            h,
            w,
            obj_logit,
            class_logit,
            anchors,
        } = &from;

        let (batch_size, _num_classes, num_anchors, height, width) = class_logit.size5()?;
        ensure!(cy.size5()? == (batch_size, 1, num_anchors, height, width),);
        ensure!(cx.size5()? == (batch_size, 1, num_anchors, height, width),);
        ensure!(h.size5()? == (batch_size, 1, num_anchors, height, width),);
        ensure!(w.size5()? == (batch_size, 1, num_anchors, height, width),);
        ensure!(obj_logit.size5()? == (batch_size, 1, num_anchors, height, width),);
        ensure!(anchors.len() == num_anchors as usize);

        Ok(Self { inner: from })
    }
}
