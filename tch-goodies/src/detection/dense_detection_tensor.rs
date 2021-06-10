use crate::{common::*, size::RatioSize};

/// Represents the output feature map of a layer.
///
/// Every belonging tensor has shape `[batch, entry, anchor, height, width]`.
#[derive(Debug, PartialEq, TensorLike, Getters)]
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

    pub fn slice_ratio(&self, y_range: Range<f64>, x_range: Range<f64>) -> Result<Self> {
        ensure!(
            (0.0..=1.0).contains(&y_range.start)
                && (0.0..=1.0).contains(&y_range.end)
                && (0.0..=1.0).contains(&x_range.start)
                && (0.0..=1.0).contains(&x_range.end)
                && y_range.end - y_range.start > 0.0
                && x_range.end - x_range.start > 0.0
        );
        let orig_h = self.height();
        let orig_w = self.width();

        let new_h = (orig_h as f64 * (y_range.end - y_range.start)).round() as i64;
        let new_w = (orig_w as f64 * (x_range.end - x_range.start)).round() as i64;

        let y_start = (y_range.start * orig_h as f64).round() as i64;
        let y_end = y_start + new_h;

        let x_start = (x_range.start * orig_w as f64).round() as i64;
        let x_end = x_start + new_w;
        self.slice(y_start..y_end, x_start..x_end)
    }

    pub fn slice(&self, y_range: Range<i64>, x_range: Range<i64>) -> Result<Self> {
        let (_, _, _, orig_h, orig_w) = self.inner.cy.size5().unwrap();
        let new_h = y_range.end - y_range.start;
        let new_w = x_range.end - x_range.start;
        ensure!(
            (0..orig_h).contains(&y_range.start)
                && (1..=orig_h).contains(&y_range.end)
                && (0..orig_w).contains(&x_range.start)
                && (1..=orig_w).contains(&x_range.end)
                && new_h > 0
                && new_w > 0
        );

        let DenseDetectionTensorUnchecked {
            cy,
            cx,
            h,
            w,
            obj_logit,
            class_logit,
            anchors,
        } = &self.inner;

        // crop
        let cy = cy
            .i((.., .., .., y_range.clone(), x_range.clone()))
            .contiguous();
        let cx = cx
            .i((.., .., .., y_range.clone(), x_range.clone()))
            .contiguous();
        let h = h
            .i((.., .., .., y_range.clone(), x_range.clone()))
            .contiguous();
        let w = w
            .i((.., .., .., y_range.clone(), x_range.clone()))
            .contiguous();
        let obj_logit = obj_logit
            .i((.., .., .., y_range.clone(), x_range.clone()))
            .contiguous();
        let class_logit = class_logit
            .i((.., .., .., y_range.clone(), x_range.clone()))
            .contiguous();

        // reposition
        let cy = (cy * orig_h as f64 - y_range.start as f64) / new_h as f64;
        let cx = (cx * orig_w as f64 - x_range.start as f64) / new_w as f64;
        let h = h * orig_h as f64 / new_h as f64;
        let w = w * orig_w as f64 / new_w as f64;

        let anchors: Vec<_> = anchors
            .iter()
            .map(|size| {
                RatioSize::new(
                    size.h() * orig_h as f64 / new_h as f64,
                    size.w() * orig_w as f64 / new_w as f64,
                )
                .unwrap()
            })
            .collect();

        let output = DenseDetectionTensorUnchecked {
            cy,
            cx,
            h,
            w,
            obj_logit,
            class_logit,
            anchors,
        }
        .try_into()
        .unwrap();

        Ok(output)
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

        let num_tensors = cy_vec.len();
        let cy_vec: Vec<_> = cy_vec
            .into_iter()
            .enumerate()
            .map(|(offset_y, cy)| (cy + offset_y as f64) / num_tensors as f64)
            .collect();
        let h_vec: Vec<_> = h_vec.into_iter().map(|h| h / num_tensors as f64).collect();

        let cy = Tensor::cat(&cy_vec, 3);
        let cx = Tensor::cat(&cx_vec, 3);
        let h = Tensor::cat(&h_vec, 3);
        let w = Tensor::cat(&w_vec, 3);
        let obj_logit = Tensor::cat(&obj_vec, 3);
        let class_logit = Tensor::cat(&class_vec, 3);
        let anchors: Vec<_> = anchors_set
            .into_iter()
            .next()
            .unwrap()
            .into_iter()
            .map(|size| RatioSize::new(size.h() / num_tensors as f64, size.w()).unwrap())
            .collect();

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

    pub fn cat_width(tensors: impl IntoIterator<Item = impl Borrow<Self>>) -> Result<Self> {
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

        let num_tensors = cx_vec.len();
        let cx_vec: Vec<_> = cx_vec
            .into_iter()
            .enumerate()
            .map(|(offset_x, cx)| (cx + offset_x as f64) / num_tensors as f64)
            .collect();
        let w_vec: Vec<_> = w_vec.into_iter().map(|w| w / num_tensors as f64).collect();

        let cy = Tensor::cat(&cy_vec, 4);
        let cx = Tensor::cat(&cx_vec, 4);
        let h = Tensor::cat(&h_vec, 4);
        let w = Tensor::cat(&w_vec, 4);
        let obj_logit = Tensor::cat(&obj_vec, 4);
        let class_logit = Tensor::cat(&class_vec, 4);
        let anchors: Vec<_> = anchors_set
            .into_iter()
            .next()
            .unwrap()
            .into_iter()
            .map(|size| RatioSize::new(size.h(), size.w() / num_tensors as f64).unwrap())
            .collect();

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
}

#[derive(Debug, PartialEq, TensorLike)]
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
    pub anchors: Vec<RatioSize<R64>>,
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
