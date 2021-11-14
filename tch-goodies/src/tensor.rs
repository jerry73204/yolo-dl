use crate::common::*;

pub use into_index_list::*;
pub use into_tensor::*;
pub use tensor_ext::*;
pub use try_into_tensor::*;

mod tensor_ext {
    use crate::Activation;

    use super::*;

    /// A trait that extends the functionality of [Tensor](tch::Tensor) type.
    pub trait TensorExt {
        fn is_all_finite(&self) -> bool;

        fn has_nan(&self) -> bool;

        fn activation(&self, act: Activation) -> Tensor;

        fn lrelu(&self) -> Tensor {
            self.leaky_relu_ext(0.2)
        }

        fn leaky_relu_ext(&self, negative_slope: impl Into<Option<f64>>) -> Tensor;

        fn f_multi_softmax(&self, dims: &[i64], kind: Kind) -> Result<Tensor>;

        fn multi_softmax(&self, dims: &[i64], kind: Kind) -> Tensor {
            self.f_multi_softmax(dims, kind).unwrap()
        }

        fn f_unfold2d(
            &self,
            kernel_size: &[i64],
            dilation: &[i64],
            padding: &[i64],
            stride: &[i64],
        ) -> Result<Tensor>;

        fn unfold2d(
            &self,
            kernel_size: &[i64],
            dilation: &[i64],
            padding: &[i64],
            stride: &[i64],
        ) -> Tensor {
            self.f_unfold2d(kernel_size, dilation, padding, stride)
                .unwrap()
        }

        fn unzip_first(&self) -> Option<Vec<Tensor>>;

        /// Reports if the tensor has zero dimension.
        fn is_empty(&self) -> bool;

        fn f_cartesian_product_nd(tensors: &[impl Borrow<Tensor>]) -> Result<Tensor>;

        fn cartesian_product_nd(tensors: &[impl Borrow<Tensor>]) -> Tensor {
            Self::f_cartesian_product_nd(tensors).unwrap()
        }

        /// Sums up a collection of tensors.
        fn f_sum_tensors<T>(tensors: impl IntoIterator<Item = T>) -> Result<Tensor>
        where
            T: Borrow<Tensor>,
        {
            let mut iter = tensors.into_iter();
            let first = iter
                .next()
                .ok_or_else(|| format_err!("the input iterator must not be empty"))?
                .borrow()
                .shallow_clone();
            let sum = iter.try_fold(first, |lhs, rhs| lhs.f_add(rhs.borrow()))?;
            Ok(sum)
        }

        /// Compute weighted sum of tensors.
        fn f_weighted_mean_tensors<T>(pairs: impl IntoIterator<Item = (T, f64)>) -> Result<Tensor>
        where
            T: Borrow<Tensor>,
        {
            let weighted_pairs: Vec<_> = pairs
                .into_iter()
                .map(|(tensor, weight)| {
                    Fallible::Ok((tensor.borrow().f_mul_scalar(weight)?, weight))
                })
                .try_collect()?;
            let (tensors, weights) = weighted_pairs.into_iter().unzip_n_vec();
            let sum_tensors = Self::f_sum_tensors(tensors)?;
            let sum_weights: f64 = weights.iter().cloned().sum();
            let mean_tensors = sum_tensors.f_div_scalar(sum_weights)?;
            Ok(mean_tensors)
        }

        /// Draw a filled rectangle on tensor.
        fn f_fill_rect_(
            &mut self,
            top: i64,
            left: i64,
            bottom: i64,
            right: i64,
            color: &Tensor,
        ) -> Result<Tensor>;

        /// Draw a filled rectangle on tensor.
        fn fill_rect_(
            &mut self,
            top: i64,
            left: i64,
            bottom: i64,
            right: i64,
            color: &Tensor,
        ) -> Tensor {
            self.f_fill_rect_(top, left, bottom, right, color).unwrap()
        }

        /// Draw a non-filled rectangle on tensor.
        fn f_draw_rect_(
            &mut self,
            top: i64,
            left: i64,
            bottom: i64,
            right: i64,
            stroke: usize,
            color: &Tensor,
        ) -> Result<Tensor>;

        /// Draw a non-filled rectangle on tensor.
        fn draw_rect_(
            &mut self,
            top: i64,
            left: i64,
            bottom: i64,
            right: i64,
            stroke: usize,
            color: &Tensor,
        ) -> Tensor {
            self.f_draw_rect_(top, left, bottom, right, stroke, color)
                .unwrap()
        }

        /// Draw filled rectangles on a tensor of a batch of images.
        fn f_batch_fill_rect_(
            &mut self,
            batches: &Tensor,
            t: &Tensor,
            l: &Tensor,
            b: &Tensor,
            r: &Tensor,
            color: &Tensor,
        ) -> Result<Tensor>;

        /// Draw filled rectangles on a tensor of a batch of images.
        fn batch_fill_rect_(
            &mut self,
            batches: &Tensor,
            t: &Tensor,
            l: &Tensor,
            b: &Tensor,
            r: &Tensor,
            color: &Tensor,
        ) -> Tensor {
            self.f_batch_fill_rect_(batches, t, l, b, r, color).unwrap()
        }

        fn f_batch_draw_rect_(
            &mut self,
            batches: &Tensor,
            t: &Tensor,
            l: &Tensor,
            b: &Tensor,
            r: &Tensor,
            stroke: usize,
            color: &Tensor,
        ) -> Result<Tensor>;

        /// Draw non-filled rectangles on a tensor of a batch of images.
        fn batch_draw_rect_(
            &mut self,
            batches: &Tensor,
            t: &Tensor,
            l: &Tensor,
            b: &Tensor,
            r: &Tensor,
            stroke: usize,
            color: &Tensor,
        ) -> Tensor {
            self.f_batch_draw_rect_(batches, t, l, b, r, stroke, color)
                .unwrap()
        }

        /// Draw filled rectangles in ratio units on a tensor of a batch of images.
        fn f_batch_fill_ratio_rect_(
            &mut self,
            batches: &Tensor,
            t: &Tensor,
            l: &Tensor,
            b: &Tensor,
            r: &Tensor,
            color: &Tensor,
        ) -> Result<Tensor>;

        /// Draw filled rectangles in ratio units on a tensor of a batch of images.
        fn batch_fill_ratio_rect_(
            &mut self,
            batches: &Tensor,
            t: &Tensor,
            l: &Tensor,
            b: &Tensor,
            r: &Tensor,
            color: &Tensor,
        ) -> Tensor {
            self.f_batch_fill_ratio_rect_(batches, t, l, b, r, color)
                .unwrap()
        }

        fn f_batch_draw_ratio_rect_(
            &mut self,
            batches: &Tensor,
            t: &Tensor,
            l: &Tensor,
            b: &Tensor,
            r: &Tensor,
            stroke: usize,
            color: &Tensor,
        ) -> Result<Tensor>;

        /// Draw non-filled rectangles in ratio units on a tensor of a batch of images.
        fn batch_draw_ratio_rect_(
            &mut self,
            batches: &Tensor,
            t: &Tensor,
            l: &Tensor,
            b: &Tensor,
            r: &Tensor,
            stroke: usize,
            color: &Tensor,
        ) -> Tensor {
            self.f_batch_draw_ratio_rect_(batches, t, l, b, r, stroke, color)
                .unwrap()
        }

        /// Crop the tensor by specifying margins.
        fn f_crop_by_ratio(&self, top: f64, left: f64, bottom: f64, right: f64) -> Result<Tensor>;

        /// Crop the tensor by specifying margins.
        fn crop_by_ratio(&self, top: f64, left: f64, bottom: f64, right: f64) -> Tensor {
            self.f_crop_by_ratio(top, left, bottom, right).unwrap()
        }

        /// Sum up a collection of tensors.
        fn sum_tensors<T>(tensors: impl IntoIterator<Item = T>) -> Tensor
        where
            T: Borrow<Tensor>,
        {
            Self::f_sum_tensors(tensors).unwrap()
        }

        /// Compute weighted sum of tensors.
        fn weighted_mean_tensors<T>(pairs: impl IntoIterator<Item = (T, f64)>) -> Tensor
        where
            T: Borrow<Tensor>,
        {
            Self::f_weighted_mean_tensors(pairs).unwrap()
        }

        /// Resize the tensor of an image and keep the ratio.
        fn resize2d(&self, new_height: i64, new_width: i64) -> Result<Tensor>;

        /// Resize the tensor of an image without keeping the ratio.
        fn resize2d_exact(&self, new_height: i64, new_width: i64) -> Result<Tensor>;

        /// Resize the tensor of an image and keep the ratio.
        fn resize2d_letterbox(&self, new_height: i64, new_width: i64) -> Result<Tensor>;

        /// Swish activation function.
        fn swish(&self) -> Tensor;

        /// Hard-Mish activation function.
        fn hard_mish(&self) -> Tensor;

        /// Convert from RGB to HSV color space.
        fn f_rgb_to_hsv(&self) -> Result<Tensor>;

        /// Convert from RGB to HSV color space.
        fn rgb_to_hsv(&self) -> Tensor {
            self.f_rgb_to_hsv().unwrap()
        }

        /// Convert from HSV to RGB color space.
        fn f_hsv_to_rgb(&self) -> Result<Tensor>;

        /// Convert from HSV to RGB color space.
        fn hsv_to_rgb(&self) -> Tensor {
            self.f_hsv_to_rgb().unwrap()
        }

        // fn normalize_channels(&self) -> Tensor;
        // fn normalize_channels_softmax(&self) -> Tensor;
    }

    impl TensorExt for Tensor {
        fn is_all_finite(&self) -> bool {
            bool::from(self.isfinite().all())
        }

        fn has_nan(&self) -> bool {
            bool::from(self.isnan().any())
        }

        fn activation(&self, act: Activation) -> Tensor {
            act.forward(self)
        }

        fn f_multi_softmax(&self, dims: &[i64], kind: Kind) -> Result<Tensor> {
            // check arguments
            let input_shape = self.size();
            let n_dims = self.dim();

            let dims_set: HashSet<_> = dims
                .iter()
                .map(|&dim_index| {
                    ensure!(
                        (0..n_dims).contains(&(dim_index as usize)),
                        "input tensor has {} dimensions, out of bound index found in {:?}",
                        n_dims,
                        dims
                    );
                    Ok(dim_index)
                })
                .try_collect()?;
            ensure!(
                dims_set.len() == dims.len(),
                "duplicated dim index found in {:?}",
                dims
            );

            let remain_dims: Vec<_> = (0..n_dims)
                .filter_map(|dim_index| {
                    let dim_index = dim_index as i64;
                    (!dims_set.contains(&dim_index)).then(|| dim_index)
                })
                .collect();

            let perm: Vec<_> = dims
                .iter()
                .cloned()
                .chain(remain_dims.iter().cloned())
                .collect();
            let inv_perm = {
                let mut inv_perm = vec![0; n_dims as usize];
                perm.iter().enumerate().for_each(|(dst, &src)| {
                    inv_perm[src as usize] = dst as i64;
                });
                inv_perm
            };

            let shape_before_softmax: Vec<_> = iter::once(-1)
                .chain(
                    remain_dims
                        .iter()
                        .map(|&dim_index| input_shape[dim_index as usize]),
                )
                .collect();
            let shape_after_softmax: Vec<_> = dims
                .iter()
                .map(|&dim_index| input_shape[dim_index as usize])
                .chain(
                    remain_dims
                        .iter()
                        .map(|&dim_index| input_shape[dim_index as usize]),
                )
                .collect();

            let output = self
                .f_permute(&perm)?
                .f_reshape(&*shape_before_softmax)?
                .f_softmax(0, kind)?
                .f_view(&*shape_after_softmax)?
                .f_permute(&inv_perm)?;

            Ok(output)
        }

        fn f_unfold2d(
            &self,
            kernel_size: &[i64],
            dilation: &[i64],
            padding: &[i64],
            stride: &[i64],
        ) -> Result<Tensor> {
            let (b, c, h, w) = self.size4().with_context(|| {
                format!(
                    "expect [batch, channel, height, width] shape, but get {:?}",
                    self.size()
                )
            })?;

            let new_h =
                (h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
            let new_w =
                (w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1;

            let output = self
                .f_im2col(kernel_size, dilation, padding, stride)?
                .f_view([b, c, kernel_size[0], kernel_size[1], new_h, new_w])?;

            Ok(output)
        }

        fn unzip_first(&self) -> Option<Vec<Tensor>> {
            let first_dim = *self.size().first()?;
            let tensors: Vec<_> = (0..first_dim).map(|index| self.select(index, 0)).collect();
            Some(tensors)
        }

        fn is_empty(&self) -> bool {
            self.numel() == 0
        }

        fn f_cartesian_product_nd(tensors: &[impl Borrow<Tensor>]) -> Result<Tensor> {
            let num_tensors = tensors.len();
            let tuples: Vec<_> = tensors
                .iter()
                .map(|tensor| -> Result<_> {
                    let tensor = tensor.borrow();
                    let shape = tensor.size();
                    let flattened = tensor.f_flatten(0, shape.len() as i64)?;
                    Ok((shape, flattened))
                })
                .try_collect()?;
            let (shapes, tensors) = tuples.into_iter().unzip_n_vec();

            let new_shape: Vec<_> = shapes
                .into_iter()
                .flatten()
                .chain(iter::once(num_tensors as i64))
                .collect();
            let output = Tensor::cartesian_prod(&tensors).view(new_shape.as_slice());
            Ok(output)
        }

        fn f_fill_rect_(
            &mut self,
            top: i64,
            left: i64,
            bottom: i64,
            right: i64,
            color: &Tensor,
        ) -> Result<Tensor> {
            tch::no_grad(|| -> Result<_> {
                match *self.size().as_slice() {
                    [_bsize, n_channels, _height, _width] => {
                        ensure!(
                            color.size1()? == n_channels,
                            "the number of channels of input and color tensors do not match"
                        );
                        let mut rect = self.i((.., .., top..bottom, left..right));
                        let expanded_color =
                            color.f_view([1, n_channels, 1, 1])?.f_expand_as(&rect)?;
                        rect.f_copy_(&expanded_color)?;
                    }
                    [n_channels, _height, _width] => {
                        ensure!(
                            color.size1()? == n_channels,
                            "the number of channels of input and color tensors do not match"
                        );
                        let mut rect = self.i((.., top..bottom, left..right));
                        let expanded_color =
                            color.f_view([n_channels, 1, 1])?.f_expand_as(&rect)?;
                        rect.f_copy_(&expanded_color)?;
                    }
                    _ => bail!("invalid shape: expect three or four dims"),
                }
                Ok(())
            })?;

            Ok(self.shallow_clone())
        }

        fn f_draw_rect_(
            &mut self,
            t: i64,
            l: i64,
            b: i64,
            r: i64,
            stroke: usize,
            color: &Tensor,
        ) -> Result<Tensor> {
            tch::no_grad(|| -> Result<_> {
                ensure!(t <= b && l <= r, "invalid tlbr parameters");

                let (n_channels, height, width) = match *self.size().as_slice() {
                    [_b, c, h, w] => (c, h, w),
                    [c, h, w] => (c, h, w),
                    _ => bail!("invalid shape: expect three or four dimensions"),
                };
                ensure!(
                    n_channels == color.size1()?,
                    "the number of channels does not match"
                );

                let half_stroke = stroke as f64 / 2.0;

                let outer_t = ((t as f64 - half_stroke) as i64).clamp(0, height - 1);
                let outer_l = ((l as f64 - half_stroke) as i64).clamp(0, width - 1);
                let outer_b = ((b as f64 + half_stroke) as i64).clamp(0, height - 1);
                let outer_r = ((r as f64 + half_stroke) as i64).clamp(0, width - 1);

                let inner_t = ((t as f64 + half_stroke) as i64).clamp(0, height - 1);
                let inner_l = ((l as f64 + half_stroke) as i64).clamp(0, width - 1);
                let inner_b = ((b as f64 - half_stroke) as i64).clamp(0, height - 1);
                let inner_r = ((r as f64 - half_stroke) as i64).clamp(0, width - 1);

                let output = self
                    .f_fill_rect_(outer_t, outer_l, inner_t, outer_r, color)?
                    .f_fill_rect_(outer_t, outer_l, outer_b, inner_l, color)?
                    .f_fill_rect_(inner_b, outer_l, outer_b, outer_r, color)?
                    .f_fill_rect_(outer_t, inner_r, outer_b, outer_r, color)?;

                Ok(output)
            })
        }

        fn f_batch_fill_rect_(
            &mut self,
            batches: &Tensor,
            t: &Tensor,
            l: &Tensor,
            b: &Tensor,
            r: &Tensor,
            color: &Tensor,
        ) -> Result<Tensor> {
            tch::no_grad(|| -> Result<_> {
                let (batch_size, n_channels, height, width) = self.size4()?;
                ensure!(
                    batches.kind() == Kind::Int64
                        && t.kind() == Kind::Int64
                        && l.kind() == Kind::Int64
                        && b.kind() == Kind::Int64
                        && r.kind() == Kind::Int64,
                    "invalid tensor kind"
                );
                ensure!(
                    n_channels == color.size1()?,
                    "number of channels does not match"
                );
                ensure!(
                    bool::from(batches.le(batch_size).all()),
                    "invalid batch index"
                );
                ensure!(
                    bool::from(t.le_tensor(b).all())
                        && bool::from(l.le_tensor(r).all())
                        && bool::from(t.ge(0).all())
                        && bool::from(t.lt(height).all())
                        && bool::from(l.ge(0).all())
                        && bool::from(l.lt(width).all())
                        && bool::from(b.ge(0).all())
                        && bool::from(b.lt(height).all())
                        && bool::from(r.ge(0).all())
                        && bool::from(r.lt(width).all()),
                    "invalid tlbr parameters"
                );
                let num_samples = batches.size1()?;
                ensure!(
                    num_samples == t.size1()?
                        && num_samples == l.size1()?
                        && num_samples == b.size1()?
                        && num_samples == r.size1()?,
                    "size mismatch"
                );

                izip!(
                    Vec::<i64>::from(batches),
                    Vec::<i64>::from(t),
                    Vec::<i64>::from(l),
                    Vec::<i64>::from(b),
                    Vec::<i64>::from(r),
                )
                .try_for_each(|(batch_index, t, l, b, r)| -> Result<_> {
                    let _ = self.i((batch_index, .., t..b, l..r)).f_copy_(
                        &color
                            .f_view([n_channels, 1, 1])?
                            .f_expand(&[n_channels, b - t, r - l], false)?,
                    )?;
                    Ok(())
                })?;

                Ok(self.shallow_clone())
            })
        }

        fn f_batch_draw_rect_(
            &mut self,
            batches: &Tensor,
            t: &Tensor,
            l: &Tensor,
            b: &Tensor,
            r: &Tensor,
            stroke: usize,
            color: &Tensor,
        ) -> Result<Tensor> {
            tch::no_grad(|| -> Result<_> {
                let (batch_size, n_channels, height, width) = self.size4()?;
                ensure!(
                    batches.kind() == Kind::Int64
                        && t.kind() == Kind::Int64
                        && l.kind() == Kind::Int64
                        && b.kind() == Kind::Int64
                        && r.kind() == Kind::Int64,
                    "invalid tensor kind"
                );
                ensure!(
                    n_channels == color.size1()?,
                    "number of channels does not match"
                );
                ensure!(
                    bool::from(batches.le(batch_size).all()),
                    "invalid batch index"
                );
                ensure!(
                    bool::from(t.le_tensor(b).all()) && bool::from(l.le_tensor(r).all()),
                    "invalid tlbr parameters"
                );
                let num_samples = batches.size1()?;
                ensure!(
                    num_samples == t.size1()?
                        && num_samples == l.size1()?
                        && num_samples == b.size1()?
                        && num_samples == r.size1()?,
                    "size mismatch"
                );

                let half_stroke = stroke as f64 / 2.0;

                let orig_t = t.to_kind(Kind::Float);
                let orig_l = l.to_kind(Kind::Float);
                let orig_b = b.to_kind(Kind::Float);
                let orig_r = r.to_kind(Kind::Float);

                let outer_t = (&orig_t - half_stroke)
                    .to_kind(Kind::Int64)
                    .clamp(0, height - 1);
                let outer_l = (&orig_l - half_stroke)
                    .to_kind(Kind::Int64)
                    .clamp(0, width - 1);
                let outer_b = (&orig_b + half_stroke)
                    .to_kind(Kind::Int64)
                    .clamp(0, height - 1);
                let outer_r = (&orig_r + half_stroke)
                    .to_kind(Kind::Int64)
                    .clamp(0, width - 1);

                let inner_t = (&orig_t + half_stroke)
                    .to_kind(Kind::Int64)
                    .clamp(0, height - 1);
                let inner_l = (&orig_l + half_stroke)
                    .to_kind(Kind::Int64)
                    .clamp(0, width - 1);
                let inner_b = (&orig_b - half_stroke)
                    .to_kind(Kind::Int64)
                    .clamp(0, height - 1);
                let inner_r = (&orig_r - half_stroke)
                    .to_kind(Kind::Int64)
                    .clamp(0, width - 1);

                let output = self
                    .f_batch_fill_rect_(batches, &outer_t, &outer_l, &inner_t, &outer_r, color)?
                    .f_batch_fill_rect_(batches, &outer_t, &outer_l, &outer_b, &inner_l, color)?
                    .f_batch_fill_rect_(batches, &inner_b, &outer_l, &outer_b, &outer_r, color)?
                    .f_batch_fill_rect_(batches, &outer_t, &inner_r, &outer_b, &outer_r, color)?;

                Ok(output)
            })
        }

        fn f_batch_fill_ratio_rect_(
            &mut self,
            batches: &Tensor,
            t: &Tensor,
            l: &Tensor,
            b: &Tensor,
            r: &Tensor,
            color: &Tensor,
        ) -> Result<Tensor> {
            let (_batch_size, _n_channels, height, width) = self.size4()?;
            let kind = self.kind();
            ensure!(
                batches.kind() == Kind::Int64
                    && t.kind() == kind
                    && l.kind() == kind
                    && b.kind() == kind
                    && r.kind() == kind,
                "invalid tensor kind"
            );

            self.f_batch_fill_rect_(
                batches,
                &(t * height as f64).to_kind(Kind::Int64),
                &(l * width as f64).to_kind(Kind::Int64),
                &(b * height as f64).to_kind(Kind::Int64),
                &(r * width as f64).to_kind(Kind::Int64),
                color,
            )
        }

        fn f_batch_draw_ratio_rect_(
            &mut self,
            batches: &Tensor,
            t: &Tensor,
            l: &Tensor,
            b: &Tensor,
            r: &Tensor,
            stroke: usize,
            color: &Tensor,
        ) -> Result<Tensor> {
            let (_batch_size, _n_channels, height, width) = self.size4()?;
            let kind = self.kind();
            ensure!(
                batches.kind() == Kind::Int64
                    && t.kind() == kind
                    && l.kind() == kind
                    && b.kind() == kind
                    && r.kind() == kind,
                "invalid tensor kind"
            );

            self.f_batch_draw_rect_(
                batches,
                &(t * height as f64).to_kind(Kind::Int64),
                &(l * width as f64).to_kind(Kind::Int64),
                &(b * height as f64).to_kind(Kind::Int64),
                &(r * width as f64).to_kind(Kind::Int64),
                stroke,
                color,
            )
        }

        fn f_crop_by_ratio(&self, top: f64, left: f64, bottom: f64, right: f64) -> Result<Tensor> {
            ensure!((0.0..=1.0).contains(&top), "invalid range");
            ensure!((0.0..=1.0).contains(&left), "invalid range");
            ensure!((0.0..=1.0).contains(&bottom), "invalid range");
            ensure!((0.0..=1.0).contains(&right), "invalid range");
            ensure!(left < right, "invalid range");
            ensure!(top < bottom, "invalid range");

            let [height, width] = match *self.size().as_slice() {
                [_c, h, w] => [h, w],
                [_b, _c, h, w] => [h, w],
                _ => bail!("input tensor must be either 3 or 4 dimensional"),
            };
            let height = height as f64;
            let width = width as f64;

            let crop_t = (top * height) as i64;
            let crop_l = (left * width) as i64;
            let crop_b = (bottom * height) as i64;
            let crop_r = (right * width) as i64;

            let cropped = match self.dim() {
                3 => self.i((.., crop_t..crop_b, crop_l..crop_r)),
                4 => self.i((.., .., crop_t..crop_b, crop_l..crop_r)),
                _ => unreachable!(),
            };

            Ok(cropped)
        }

        fn resize2d(&self, new_height: i64, new_width: i64) -> Result<Tensor> {
            tch::no_grad(|| match (self.kind(), self.size().as_slice()) {
                (Kind::Float, &[_n_channels, _height, _width]) => {
                    let resized = vision::image::resize_preserve_aspect_ratio(
                        &(self * 255.0).to_kind(Kind::Uint8),
                        new_width,
                        new_height,
                    )?
                    .to_kind(Kind::Float)
                        / 255.0;
                    Ok(resized)
                }
                (Kind::Uint8, &[_n_channels, _height, _width]) => {
                    let resized =
                        vision::image::resize_preserve_aspect_ratio(self, new_width, new_height)?;
                    Ok(resized)
                }
                (_, &[_n_channels, _height, _width]) => bail!("unsupported data kind"),
                (Kind::Float, &[batch_size, _n_channels, _height, _width]) => {
                    let self_scaled = (self * 255.0).to_kind(Kind::Uint8);
                    let resized_vec: Vec<_> = (0..batch_size)
                        .map(|index| -> Result<_> {
                            let resized = vision::image::resize_preserve_aspect_ratio(
                                &self_scaled.select(0, index),
                                new_width,
                                new_height,
                            )?;
                            Ok(resized)
                        })
                        .try_collect()?;
                    let resized = Tensor::stack(resized_vec.as_slice(), 0);
                    let resized_scaled = resized.to_kind(Kind::Float) / 255.0;
                    Ok(resized_scaled)
                }
                (Kind::Uint8, &[batch_size, _n_channels, _height, _width]) => {
                    let resized_vec: Vec<_> = (0..batch_size)
                        .map(|index| -> Result<_> {
                            let resized = vision::image::resize_preserve_aspect_ratio(
                                &self.select(0, index),
                                new_width,
                                new_height,
                            )?;
                            Ok(resized)
                        })
                        .try_collect()?;
                    let resized = Tensor::stack(resized_vec.as_slice(), 0);
                    Ok(resized)
                }
                (_, &[_batch_size, _n_channels, _height, _width]) => bail!("unsupported data kind"),
                _ => bail!("invalid shape: expect three or four dimensions"),
            })
        }

        fn resize2d_exact(&self, new_height: i64, new_width: i64) -> Result<Tensor> {
            tch::no_grad(|| match (self.kind(), self.size().as_slice()) {
                (Kind::Uint8, &[_n_channels, _height, _width]) => {
                    let resized = vision::image::resize(self, new_width, new_height)?;
                    Ok(resized)
                }
                (Kind::Float, &[_n_channels, _height, _width]) => {
                    let resized = vision::image::resize(
                        &(self * 255.0).to_kind(Kind::Uint8),
                        new_width,
                        new_height,
                    )?
                    .to_kind(Kind::Float)
                        / 255.0;
                    Ok(resized)
                }
                (_, &[_n_channels, _height, _width]) => bail!("unsupported data kind"),
                (Kind::Float, &[batch_size, _n_channels, _height, _width]) => {
                    let self_scaled = (self * 255.0).to_kind(Kind::Uint8);
                    let resized_vec: Vec<_> = (0..batch_size)
                        .map(|index| -> Result<_> {
                            let resized = vision::image::resize(
                                &self_scaled.select(0, index),
                                new_width,
                                new_height,
                            )?;
                            Ok(resized)
                        })
                        .try_collect()?;
                    let resized = Tensor::stack(resized_vec.as_slice(), 0);
                    let resized_scaled = resized.to_kind(Kind::Float) / 255.0;
                    Ok(resized_scaled)
                }
                (Kind::Uint8, &[batch_size, _n_channels, _height, _width]) => {
                    let resized_vec: Vec<_> = (0..batch_size)
                        .map(|index| -> Result<_> {
                            let resized = vision::image::resize(
                                &self.select(0, index),
                                new_width,
                                new_height,
                            )?;
                            Ok(resized)
                        })
                        .try_collect()?;
                    let resized = Tensor::stack(resized_vec.as_slice(), 0);
                    Ok(resized)
                }
                (_, &[_batch_size, _n_channels, _height, _width]) => bail!("unsupported data kind"),
                _ => bail!("invalid shape: expect three or four dimensions"),
            })
        }

        fn resize2d_letterbox(&self, new_height: i64, new_width: i64) -> Result<Tensor> {
            let inner_rect = |height: i64, width: i64| {
                let scale_h = new_height as f64 / height as f64;
                let scale_w = new_width as f64 / width as f64;
                let (inner_h, inner_w) = if scale_h <= scale_w {
                    (
                        new_height,
                        (width as f64 * new_height as f64 / height as f64) as i64,
                    )
                } else {
                    (
                        (height as f64 * new_width as f64 / width as f64) as i64,
                        new_width,
                    )
                };
                let (top, left) = ((new_height - inner_h) / 2, (new_width - inner_w) / 2);
                (inner_h, inner_w, top, left)
            };

            tch::no_grad(|| match (self.kind(), self.size().as_slice()) {
                (Kind::Uint8, &[channels, height, width]) => {
                    let (inner_h, inner_w, top, left) = inner_rect(height, width);
                    let inner = vision::image::resize(self, inner_w, inner_h)?;
                    let outer = Tensor::zeros(
                        &[channels, new_height, new_width],
                        (self.kind(), self.device()),
                    );
                    outer
                        .narrow(1, top, inner_h)
                        .narrow(2, left, inner_w)
                        .copy_(&inner);
                    Ok(outer)
                }
                (Kind::Float, &[channels, height, width]) => {
                    let (inner_h, inner_w, top, left) = inner_rect(height, width);
                    let inner = vision::image::resize(
                        &(self * 255.0).to_kind(Kind::Uint8),
                        inner_w,
                        inner_h,
                    )?
                    .to_kind(Kind::Float)
                        / 255.0;
                    let outer = Tensor::zeros(
                        &[channels, new_height, new_width],
                        (self.kind(), self.device()),
                    );
                    outer
                        .narrow(1, top, inner_h)
                        .narrow(2, left, inner_w)
                        .copy_(&inner);
                    Ok(outer)
                }
                (_, &[_n_channels, _height, _width]) => bail!("unsupported data kind"),
                (Kind::Float, &[batch_size, channels, height, width]) => {
                    let (inner_h, inner_w, top, left) = inner_rect(height, width);
                    let scaled = (self * 255.0).to_kind(Kind::Uint8);
                    let resized_vec: Vec<_> = (0..batch_size)
                        .map(|index| {
                            vision::image::resize(&scaled.select(0, index), inner_w, inner_h)
                        })
                        .try_collect()?;
                    let inner =
                        Tensor::stack(resized_vec.as_slice(), 0).to_kind(Kind::Float) / 255.0;
                    let outer = Tensor::zeros(
                        &[batch_size, channels, new_height, new_width],
                        (self.kind(), self.device()),
                    );
                    outer
                        .narrow(2, top, inner_h)
                        .narrow(3, left, inner_w)
                        .copy_(&inner);
                    Ok(outer)
                }
                (Kind::Uint8, &[batch_size, channels, height, width]) => {
                    let (inner_h, inner_w, top, left) = inner_rect(height, width);
                    let resized_vec: Vec<_> = (0..batch_size)
                        .map(|index| -> Result<_> {
                            let resized =
                                vision::image::resize(&self.select(0, index), inner_w, inner_h)?;
                            Ok(resized)
                        })
                        .try_collect()?;
                    let inner = Tensor::stack(resized_vec.as_slice(), 0);
                    let outer = Tensor::zeros(
                        &[batch_size, channels, new_height, new_width],
                        (self.kind(), self.device()),
                    );
                    outer
                        .narrow(2, top, inner_h)
                        .narrow(3, left, inner_w)
                        .copy_(&inner);
                    Ok(outer)
                }
                (_, &[_batch_size, _n_channels, _height, _width]) => bail!("unsupported data kind"),
                _ => bail!("invalid shape: expect three or four dimensions"),
            })
        }

        fn swish(&self) -> Tensor {
            self * self.sigmoid()
        }

        fn hard_mish(&self) -> Tensor {
            let case1 = self.clamp(-2.0, 0.0);
            let case2 = self.clamp_min(0.0);
            (case1.pow(&2i64.into()) / 2.0 + &case1) + case2
        }

        // fn normalize_channels(&self) -> Tensor {
        //     todo!();
        // }

        // fn normalize_channels_softmax(&self) -> Tensor {
        //     todo!();
        // }

        fn f_rgb_to_hsv(&self) -> Result<Tensor> {
            let eps = 1e-10;
            let rgb = self;

            let (channel_index, channels) = match rgb.size().as_slice() {
                &[c, _h, _w] => (0, c),
                &[_b, c, _h, _w] => (1, c),
                dims => bail!("expect 3 or 4 dimensions, but get {:?}", dims),
            };

            ensure!(
                channels == 3,
                "channel size must be 3, but get {}",
                channels
            );

            let red = rgb.select(channel_index, 0);
            let green = rgb.select(channel_index, 1);
            let blue = rgb.select(channel_index, 2);

            let (max, argmax) = rgb.max_dim(channel_index, false);
            let (min, _argmin) = rgb.min_dim(channel_index, false);
            let diff = &max - &min;

            let value = max;
            let saturation = (&diff / &value).where_self(&value.gt(eps), &value.zeros_like());

            let case1 = value.zeros_like();
            let case2 = (&green - &blue) / &diff;
            let case3 = (&blue - &red) / &diff + 2.0;
            let case4 = (&red - &green) / &diff + 4.0;

            let hue = {
                let hue = case1.where_self(
                    &diff.le(eps),
                    &case2.where_self(&argmax.eq(0), &case3.where_self(&argmax.eq(1), &case4)),
                );
                (hue + 6.0).fmod(6.0) / 6.0
            };

            let hsv = Tensor::stack(&[hue, saturation, value], channel_index);

            debug_assert!(
                !bool::from(hsv.isnan().any()),
                "NaN detected in RGB to HSV conversion"
            );

            Ok(hsv)
        }

        fn f_hsv_to_rgb(&self) -> Result<Tensor> {
            let hsv = self;
            let (channel_index, channels) = match hsv.size().as_slice() {
                &[c, _h, _w] => (0, c),
                &[_b, c, _h, _w] => (1, c),
                dims => bail!("expect 3 or 4 dimensions, but get {:?}", dims),
            };
            ensure!(
                channels == 3,
                "channel size must be 3, but get {}",
                channels
            );

            let hue = hsv.select(channel_index, 0);
            let saturation = hsv.select(channel_index, 1);
            let value = hsv.select(channel_index, 2);

            let func = |n: i64| {
                let k = (&hue * 6.0 + n as f64).fmod(6.0);
                &value * (1.0 - &saturation * k.min_other(&(-&k + 4.0)).clamp(0.0, 1.0))
            };

            let red = func(5);
            let green = func(3);
            let blue = func(1);
            let rgb = Tensor::stack(&[red, green, blue], channel_index);

            Ok(rgb)
        }

        fn leaky_relu_ext(&self, negative_slope: impl Into<Option<f64>>) -> Tensor {
            self.maximum(&(self * negative_slope.into().unwrap_or(0.01)))
        }
    }
}

mod into_tensor {
    use super::*;

    pub trait IntoTensor {
        fn into_tensor(self) -> Tensor;
    }

    impl<P, Container> IntoTensor for &ImageBuffer<P, Container>
    where
        P: Pixel + 'static,
        P::Subpixel: 'static + Element,
        Container: Deref<Target = [P::Subpixel]>,
    {
        fn into_tensor(self) -> Tensor {
            let (width, height) = self.dimensions();
            let height = height as usize;
            let width = width as usize;
            let channels = P::CHANNEL_COUNT as usize;

            let buffer = unsafe {
                let buf_len = channels * height * width;
                let mut buffer: Vec<P::Subpixel> = Vec::with_capacity(buf_len);
                let ptr = buffer.as_mut_ptr();
                self.enumerate_pixels().for_each(|(x, y, pixel)| {
                    let x = x as usize;
                    let y = y as usize;
                    pixel
                        .channels()
                        .iter()
                        .cloned()
                        .enumerate()
                        .for_each(|(c, component)| {
                            *ptr.add(x + width * (y + height * c)) = component;
                        });
                });
                buffer.set_len(buf_len);
                buffer
            };

            Tensor::of_slice(&buffer).view([channels as i64, height as i64, width as i64])
        }
    }
}

mod try_into_tensor {
    use super::*;

    pub trait TryIntoTensor {
        type Error;

        fn try_into_tensor(self) -> Result<Tensor, Self::Error>;
    }

    impl TryIntoTensor for &DynamicImage {
        type Error = Error;

        fn try_into_tensor(self) -> Result<Tensor, Self::Error> {
            let tensor = match self {
                DynamicImage::ImageLuma8(image) => image.into_tensor(),
                DynamicImage::ImageLumaA8(image) => image.into_tensor(),
                DynamicImage::ImageRgb8(image) => image.into_tensor(),
                DynamicImage::ImageRgba8(image) => image.into_tensor(),
                DynamicImage::ImageBgr8(image) => image.into_tensor(),
                DynamicImage::ImageBgra8(image) => image.into_tensor(),
                _ => bail!("cannot convert an image with u16 components to a tensor"),
            };
            Ok(tensor)
        }
    }
}

mod into_index_list {
    use super::*;

    pub trait IntoTensorIndex {
        fn to_tensor_index(&self, device: Device) -> Option<Tensor>;
    }

    impl IntoTensorIndex for Tensor {
        fn to_tensor_index(&self, device: Device) -> Option<Tensor> {
            Some(self.to_device(device))
        }
    }

    impl IntoTensorIndex for &Tensor {
        fn to_tensor_index(&self, device: Device) -> Option<Tensor> {
            Some((*self).to_device(device))
        }
    }

    impl IntoTensorIndex for Option<Tensor> {
        fn to_tensor_index(&self, device: Device) -> Option<Tensor> {
            self.as_ref().map(|tensor| tensor.to_device(device))
        }
    }

    impl IntoTensorIndex for Option<&Tensor> {
        fn to_tensor_index(&self, device: Device) -> Option<Tensor> {
            self.map(|tensor| tensor.to_device(device))
        }
    }
    impl IntoTensorIndex for &Option<Tensor> {
        fn to_tensor_index(&self, device: Device) -> Option<Tensor> {
            self.as_ref().map(|tensor| tensor.to_device(device))
        }
    }
    impl IntoTensorIndex for &Option<&Tensor> {
        fn to_tensor_index(&self, device: Device) -> Option<Tensor> {
            self.map(|tensor| tensor.to_device(device))
        }
    }
    impl IntoTensorIndex for &[i64] {
        fn to_tensor_index(&self, device: Device) -> Option<Tensor> {
            Some(Tensor::of_slice(self).to_device(device))
        }
    }
    impl IntoTensorIndex for Vec<i64> {
        fn to_tensor_index(&self, device: Device) -> Option<Tensor> {
            Some(Tensor::of_slice(self.as_slice()).to_device(device))
        }
    }
    impl<const LEN: usize> IntoTensorIndex for [i64; LEN] {
        fn to_tensor_index(&self, device: Device) -> Option<Tensor> {
            Some(Tensor::of_slice(self.as_ref()).to_device(device))
        }
    }
    impl<const LEN: usize> IntoTensorIndex for &[i64; LEN] {
        fn to_tensor_index(&self, device: Device) -> Option<Tensor> {
            Some(Tensor::of_slice(self.as_ref()).to_device(device))
        }
    }

    pub trait IntoIndexList {
        fn into_index_list(self, device: Device) -> Vec<Option<Tensor>>;
    }

    // slice
    impl<T> IntoIndexList for &[T]
    where
        T: IntoTensorIndex,
    {
        fn into_index_list(self, device: Device) -> Vec<Option<Tensor>> {
            self.iter()
                .map(|index| index.to_tensor_index(device))
                .collect()
        }
    }

    // vec
    impl<T> IntoIndexList for Vec<T>
    where
        T: IntoTensorIndex,
    {
        fn into_index_list(self, device: Device) -> Vec<Option<Tensor>> {
            self.into_iter()
                .map(|index| index.to_tensor_index(device))
                .collect()
        }
    }

    // tuple
    macro_rules! impl_tuple {
        ( $(($input_ty:ident, $input_arg:ident)),* ) => {
            impl< $($input_ty),* > IntoIndexList for ( $($input_ty,)* )
            where
                $($input_ty: IntoTensorIndex),*
            {
                fn into_index_list(self, device: Device) -> Vec<Option<Tensor>> {
                    let ($($input_arg,)*) = self;
                    [ $( <$input_ty as IntoTensorIndex>::to_tensor_index(& $input_arg, device) ),* ]
                        .into_index_list(device)
                }
            }
        };
    }

    impl_tuple!((T0, arg0));
    impl_tuple!((T0, arg0), (T1, arg1));
    impl_tuple!((T0, arg0), (T1, arg1), (T2, arg2));
    impl_tuple!((T0, arg0), (T1, arg1), (T2, arg2), (T3, arg3));
    impl_tuple!((T0, arg0), (T1, arg1), (T2, arg2), (T3, arg3), (T4, arg4));
    impl_tuple!(
        (T0, arg0),
        (T1, arg1),
        (T2, arg2),
        (T3, arg3),
        (T4, arg4),
        (T5, arg5)
    );
    impl_tuple!(
        (T0, arg0),
        (T1, arg1),
        (T2, arg2),
        (T3, arg3),
        (T4, arg4),
        (T5, arg5),
        (T6, arg6)
    );
    impl_tuple!(
        (T0, arg0),
        (T1, arg1),
        (T2, arg2),
        (T3, arg3),
        (T4, arg4),
        (T5, arg5),
        (T6, arg6),
        (T7, arg7)
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_convert::TryIntoCv;
    use itertools::iproduct;
    use ndarray::{Array4, Array6};
    use tch::kind::FLOAT_CPU;

    #[test]
    fn multi_softmax_test() {
        let input = Tensor::rand(&[3, 5, 2, 8, 6, 7, 2], FLOAT_CPU);
        let output = input.multi_softmax(&[1, 2, 4], Kind::Float);
        assert_eq!(input.size(), output.size());

        let sum = output.sum_dim_intlist(&[1, 2, 4], false, Kind::Float);

        assert!(bool::from(
            (&sum - Tensor::from(1f32).view([1, 1, 1, 1]).expand_as(&sum))
                .abs()
                .lt(1e-6)
                .all()
        ));
        assert!(bool::from(output.ge(0.0).all()) && bool::from(output.le(1.0).all()));
    }

    #[test]
    fn unfold2d_test() {
        let b = 16;
        let c = 3;
        let h = 10;
        let w = 11;
        let ky = 5;
        let kx = 3;

        let input = Tensor::rand(&[b, c, h, w], FLOAT_CPU);
        let output = input.unfold2d(&[ky, kx], &[1, 1], &[ky / 2, kx / 2], &[1, 1]);

        assert_eq!(output.size(), vec![b, c, ky, kx, h, w]);

        let input_array: Array4<f32> = input.try_into_cv().unwrap();
        let output_array: Array6<f32> = output.try_into_cv().unwrap();

        let mut expect_array = Array6::<f32>::zeros([
            b as usize,
            c as usize,
            ky as usize,
            kx as usize,
            h as usize,
            w as usize,
        ]);

        iproduct!(0..b, 0..c, 0..h, 0..w, 0..ky, 0..kx).for_each(|args| {
            let (bi, ci, hi, wi, kyi, kxi) = args;
            let i = hi + kyi - ky / 2;
            let j = wi + kxi - kx / 2;

            if (0..h).contains(&i) && (0..w).contains(&j) {
                expect_array[[
                    bi as usize,
                    ci as usize,
                    kyi as usize,
                    kxi as usize,
                    hi as usize,
                    wi as usize,
                ]] = input_array[[bi as usize, ci as usize, i as usize, j as usize]];
            }
        });

        assert_eq!(expect_array, output_array);
    }

    #[test]
    fn hue_rgb_conv() {
        const EPSILON: f64 = 1e-5;

        for _ in 0..100 {
            let input = Tensor::rand(&[3, 512, 512], FLOAT_CPU);
            let output = input.rgb_to_hsv().hsv_to_rgb();
            assert!(
                <f32 as From<_>>::from(output.min()) >= 0.0
                    && <f32 as From<_>>::from(output.max()) <= 1.0
            );

            let diff = (input - output).abs();
            assert!(bool::from(diff.le(EPSILON).all()));
        }
    }
}
