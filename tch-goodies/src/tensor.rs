use crate::common::*;

pub use into_index_list::*;
pub use into_tensor::*;
pub use tensor_ext::*;
pub use try_into_tensor::*;

mod tensor_ext {
    use super::*;

    /// A trait that extends the functionality of [Tensor](tch::Tensor) type.
    pub trait TensorExt {
        fn f_index_put_opt_(
            &mut self,
            indexes: impl IntoIndexList,
            values: &Tensor,
            accumulate: bool,
        ) -> Result<Tensor>;

        fn index_put_opt_(
            &mut self,
            indexes: impl IntoIndexList,
            values: &Tensor,
            accumulate: bool,
        ) -> Tensor {
            self.f_index_put_opt_(indexes, values, accumulate).unwrap()
        }

        fn f_index_opt(&self, indexes: impl IntoIndexList) -> Result<Tensor>;

        fn index_opt(&self, indexes: impl IntoIndexList) -> Tensor {
            self.f_index_opt(indexes).unwrap()
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
                .map(|(tensor, weight)| Fallible::Ok((tensor.borrow().f_mul1(weight)?, weight)))
                .try_collect()?;
            let (tensors, weights) = weighted_pairs.into_iter().unzip_n_vec();
            let sum_tensors = Self::f_sum_tensors(tensors)?;
            let sum_weights: f64 = weights.iter().cloned().sum();
            let mean_tensors = sum_tensors.f_div1(sum_weights)?;
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
        fn f_batch_fill_rect_(&mut self, btlbrs: &[[i64; 5]], color: &Tensor) -> Result<Tensor>;

        /// Draw filled rectangles on a tensor of a batch of images.
        fn batch_fill_rect_(&mut self, btlbrs: &[[i64; 5]], color: &Tensor) -> Tensor {
            self.f_batch_fill_rect_(btlbrs, color).unwrap()
        }

        fn f_batch_draw_rect_(
            &mut self,
            btlbrs: &[[i64; 5]],
            stroke: usize,
            color: &Tensor,
        ) -> Result<Tensor>;

        /// Draw non-filled rectangles on a tensor of a batch of images.
        fn batch_draw_rect_(
            &mut self,
            btlbrs: &[[i64; 5]],
            stroke: usize,
            color: &Tensor,
        ) -> Tensor {
            self.f_batch_draw_rect_(btlbrs, stroke, color).unwrap()
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

        /// Hard-Swish activation function.
        fn hard_swish(&self) -> Tensor;

        /// Mish activation function.
        fn mish(&self) -> Tensor;

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
        fn f_index_put_opt_(
            &mut self,
            indexes: impl IntoIndexList,
            values: &Tensor,
            accumulate: bool,
        ) -> Result<Tensor> {
            let device = self.device();
            let indexes = indexes.into_index_list(device);

            let occurrences: Vec<_> = indexes
                .iter()
                .enumerate()
                .filter_map(|(index, tensor)| tensor.as_ref().map(|_| index))
                .collect();

            let perm: Vec<_> = occurrences
                .iter()
                .cloned()
                .chain(
                    indexes
                        .iter()
                        .enumerate()
                        .filter_map(|(index, tensor)| match tensor {
                            Some(_) => None,
                            None => Some(index),
                        }),
                )
                .map(|index| index as i64)
                .collect();

            let output = {
                let tch_indexes: Vec<_> = occurrences
                    .iter()
                    .map(|&index| indexes[index].as_ref().unwrap())
                    .collect();
                self.f_permute(&perm)?
                    .f_index_put_(&tch_indexes, values, accumulate)?
            };

            Ok(output)
        }

        fn f_index_opt(&self, indexes: impl IntoIndexList) -> Result<Tensor> {
            let device = self.device();
            let indexes = indexes.into_index_list(device);

            let occurrences: Vec<_> = indexes
                .iter()
                .enumerate()
                .filter_map(|(index, tensor)| tensor.as_ref().map(|_| index))
                .collect();

            let perm: Vec<_> = occurrences
                .iter()
                .cloned()
                .chain(
                    indexes
                        .iter()
                        .enumerate()
                        .filter_map(|(index, tensor)| match tensor {
                            Some(_) => None,
                            None => Some(index),
                        }),
                )
                .map(|index| index as i64)
                .collect();

            let output = {
                let tch_indexes: Vec<_> = occurrences
                    .iter()
                    .map(|&index| indexes[index].as_ref().unwrap())
                    .collect();
                self.f_permute(&perm)?.f_index(&tch_indexes)?
            };

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
                match self.size().as_slice() {
                    &[_bsize, n_channels, _height, _width] => {
                        ensure!(
                            color.size1()? == n_channels,
                            "the number of channels of input and color tensors do not match"
                        );
                        let mut rect = self.i((.., .., top..bottom, left..right));
                        let expanded_color =
                            color.f_view([1, n_channels, 1, 1])?.f_expand_as(&rect)?;
                        rect.f_copy_(&expanded_color)?;
                    }
                    &[n_channels, _height, _width] => {
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
            let (n_channels, height, width) = match self.size().as_slice() {
                &[_b, c, h, w] => (c, h, w),
                &[c, h, w] => (c, h, w),
                _ => bail!("invalid shape: expect three or four dimensions"),
            };
            ensure!(
                n_channels == color.size1()?,
                "the number of channels does not match"
            );

            let (outer_t, outer_l, outer_b, outer_r) = {
                let half_stroke = (stroke / 2) as i64;
                (
                    t - half_stroke,
                    l - half_stroke,
                    b + half_stroke,
                    r + half_stroke,
                )
            };
            let (inner_t, inner_l, inner_b, inner_r) = {
                let stroke = stroke as i64;
                (
                    outer_t + stroke,
                    outer_l + stroke,
                    outer_b - stroke,
                    outer_r - stroke,
                )
            };

            let outer_t = outer_t.max(0).min(height - 1);
            let outer_b = outer_b.max(0).min(height - 1);
            let outer_l = outer_l.max(0).min(width - 1);
            let outer_r = outer_r.max(0).min(width - 1);

            let inner_t = inner_t.max(0).min(height - 1);
            let inner_b = inner_b.max(0).min(height - 1);
            let inner_l = inner_l.max(0).min(width - 1);
            let inner_r = inner_r.max(0).min(width - 1);

            tch::no_grad(|| -> Result<_> {
                // draw t edge
                let _ = self.f_fill_rect_(outer_t, outer_l, inner_t, outer_r, color)?;

                // draw l edge
                let _ = self.f_fill_rect_(outer_t, outer_l, outer_b, inner_l, color)?;

                // draw b edge
                let _ = self.f_fill_rect_(inner_b, outer_l, outer_b, outer_r, color)?;

                // draw r edge
                let _ = self.f_fill_rect_(outer_t, inner_r, outer_b, outer_r, color)?;

                Ok(())
            })?;

            Ok(self.shallow_clone())
        }

        fn f_batch_fill_rect_(&mut self, btlbrs: &[[i64; 5]], color: &Tensor) -> Result<Tensor> {
            let (batch_size, n_channels, height, width) = self.size4()?;
            ensure!(
                n_channels == color.size1()?,
                "number of channels does not match"
            );

            tch::no_grad(|| -> Result<_> {
                for [batch_index, t, l, b, r] in btlbrs.iter().cloned() {
                    ensure!(
                0 <= batch_index
                    && batch_index < batch_size
                    && 0 <= t
                    && t <= b
                    && b < height
                    && 0 <= l
                    && l <= r
                    && r < width,
                "invalid bounding box coordinates: btlbr = {:?}, batch_size = {}, height = {}, width = {}",
                [batch_index, t, l, b, r], batch_size, height, width
            );

                    let _ = self.i((batch_index, .., t..b, l..r)).f_copy_(
                        &color
                            .f_view([n_channels, 1, 1])?
                            .f_expand(&[n_channels, b - t, r - l], false)?,
                    )?;
                }
                Ok(())
            })?;

            Ok(self.shallow_clone())
        }

        fn f_batch_draw_rect_(
            &mut self,
            btlbrs: &[[i64; 5]],
            stroke: usize,
            color: &Tensor,
        ) -> Result<Tensor> {
            let (batch_size, n_channels, height, width) = self.size4()?;
            ensure!(
                n_channels == color.size1()?,
                "number of channels does not match"
            );
            for [batch_index, top, left, bottom, right] in btlbrs.iter().cloned() {
                ensure!(
                    0 <= batch_index
                        && batch_index < batch_size
                        && 0 <= top
                        && top <= bottom
                        && bottom < height
                        && 0 <= left
                        && left <= right
                        && right < width,
                    "invalid bounding box coordinates {:?}",
                    [batch_index, top, left, bottom, right]
                );
            }

            let (bboxes_t, bboxes_l, bboxes_b, bboxes_r) = btlbrs
                .iter()
                .cloned()
                .map(|[batch_index, top, left, bottom, right]| {
                    let [outer_t, outer_l, outer_b, outer_r] = {
                        let half_stroke = stroke as f64 / 2.0;
                        [
                            (top as f64 - half_stroke) as i64,
                            (left as f64 - half_stroke) as i64,
                            (bottom as f64 + half_stroke) as i64,
                            (right as f64 + half_stroke) as i64,
                        ]
                    };
                    let [inner_t, inner_l, inner_b, inner_r] = {
                        let stroke = stroke as i64;
                        [
                            (outer_t as f64 + stroke as f64) as i64,
                            (outer_l as f64 + stroke as f64) as i64,
                            (outer_b as f64 - stroke as f64) as i64,
                            (outer_r as f64 - stroke as f64) as i64,
                        ]
                    };

                    let outer_t = outer_t.max(0).min(height - 1);
                    let outer_b = outer_b.max(0).min(height - 1);
                    let outer_l = outer_l.max(0).min(width - 1);
                    let outer_r = outer_r.max(0).min(width - 1);

                    let inner_t = inner_t.max(0).min(height - 1);
                    let inner_b = inner_b.max(0).min(height - 1);
                    let inner_l = inner_l.max(0).min(width - 1);
                    let inner_r = inner_r.max(0).min(width - 1);

                    let bbox_t = [batch_index, outer_t, outer_l, inner_t, outer_r];
                    let bbox_l = [batch_index, outer_t, outer_l, outer_b, inner_l];
                    let bbox_b = [batch_index, inner_b, outer_l, outer_b, outer_r];
                    let bbox_r = [batch_index, outer_t, inner_r, outer_b, outer_r];

                    (bbox_t, bbox_l, bbox_b, bbox_r)
                })
                .unzip_n_vec();

            tch::no_grad(|| -> Result<_> {
                let _ = self.f_batch_fill_rect_(&bboxes_t, color)?;
                let _ = self.f_batch_fill_rect_(&bboxes_l, color)?;
                let _ = self.f_batch_fill_rect_(&bboxes_b, color)?;
                let _ = self.f_batch_fill_rect_(&bboxes_r, color)?;
                Ok(())
            })?;

            Ok(self.shallow_clone())
        }

        fn f_crop_by_ratio(&self, top: f64, left: f64, bottom: f64, right: f64) -> Result<Tensor> {
            ensure!((0.0..=1.0).contains(&top), "invalid range");
            ensure!((0.0..=1.0).contains(&left), "invalid range");
            ensure!((0.0..=1.0).contains(&bottom), "invalid range");
            ensure!((0.0..=1.0).contains(&right), "invalid range");
            ensure!(left < right, "invalid range");
            ensure!(top < bottom, "invalid range");

            let [height, width] = match self.size().as_slice() {
                &[_c, h, w] => [h, w],
                &[_b, _c, h, w] => [h, w],
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

        fn hard_swish(&self) -> Tensor {
            self * (self + 3.0).clamp(0.0, 6.0) / 6.0
        }

        fn mish(&self) -> Tensor {
            self * &self.softplus().tanh()
        }

        fn hard_mish(&self) -> Tensor {
            let case1 = self.clamp(-2.0, 0.0);
            let case2 = self.clamp_min(0.0);
            (case1.pow(2.0) / 2.0 + &case1) + case2
        }

        // fn normalize_channels(&self) -> Tensor {
        //     todo!();
        // }

        // fn normalize_channels_softmax(&self) -> Tensor {
        //     todo!();
        // }

        fn f_rgb_to_hsv(&self) -> Result<Tensor> {
            let eps = 1e-4;
            let rgb = self;
            let (channels, _height, _width) = rgb.size3()?;
            ensure!(
                channels == 3,
                "channel size must be 3, but get {}",
                channels
            );

            let red = rgb.select(0, 0);
            let green = rgb.select(0, 1);
            let blue = rgb.select(0, 2);

            let (max, argmax) = rgb.max2(0, false);
            let (min, _argmin) = rgb.min2(0, false);
            let diff = &max - &min;

            let value = max;
            let saturation = (&diff / &value).where1(&value.gt(eps), &value.zeros_like());

            let case1 = value.zeros_like();
            let case2 = (&green - &blue) / &diff;
            let case3 = (&blue - &red) / &diff + 2.0;
            let case4 = (&red - &green) / &diff + 4.0;

            let hue = {
                let hue = case1.where1(
                    &diff.le(eps),
                    &case2.where1(&argmax.eq(0), &case3.where1(&argmax.eq(1), &case4)),
                );
                let hue = hue.where1(&hue.ge(0.0), &(&hue + 6.0));
                let hue = hue / 6.0;
                hue
            };

            let hsv = Tensor::stack(&[hue, saturation, value], 0);

            debug_assert!(
                !bool::from(hsv.isnan().any()),
                "NaN detected in RGB to HSV conversion"
            );

            Ok(hsv)
        }

        fn f_hsv_to_rgb(&self) -> Result<Tensor> {
            let hsv = self;
            let (channels, _height, _width) = hsv.size3()?;
            ensure!(
                channels == 3,
                "channel size must be 3, but get {}",
                channels
            );

            let hue = hsv.select(0, 0);
            let saturation = hsv.select(0, 1);
            let value = hsv.select(0, 2);

            let func = |n: i64| {
                let k = (&hue + n as f64).fmod(2);
                &value
                    - &value
                        * &saturation
                        * value
                            .zeros_like()
                            .max1(&k.min1(&(-&k + 4.0)).clamp_min(1.0))
            };

            let red = func(5);
            let green = func(3);
            let blue = func(1);
            let rgb = Tensor::stack(&[red, green, blue], 0);

            Ok(rgb)
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
        fn into_tensor_index(&self, device: Device) -> Option<Tensor>;
    }

    impl IntoTensorIndex for Tensor {
        fn into_tensor_index(&self, device: Device) -> Option<Tensor> {
            Some(self.to_device(device))
        }
    }

    impl IntoTensorIndex for &Tensor {
        fn into_tensor_index(&self, device: Device) -> Option<Tensor> {
            Some((*self).to_device(device))
        }
    }

    impl IntoTensorIndex for Option<Tensor> {
        fn into_tensor_index(&self, device: Device) -> Option<Tensor> {
            self.as_ref().map(|tensor| tensor.to_device(device))
        }
    }

    impl IntoTensorIndex for Option<&Tensor> {
        fn into_tensor_index(&self, device: Device) -> Option<Tensor> {
            self.map(|tensor| tensor.to_device(device))
        }
    }
    impl IntoTensorIndex for &Option<Tensor> {
        fn into_tensor_index(&self, device: Device) -> Option<Tensor> {
            self.as_ref().map(|tensor| tensor.to_device(device))
        }
    }
    impl IntoTensorIndex for &Option<&Tensor> {
        fn into_tensor_index(&self, device: Device) -> Option<Tensor> {
            self.map(|tensor| tensor.to_device(device))
        }
    }
    impl IntoTensorIndex for &[i64] {
        fn into_tensor_index(&self, device: Device) -> Option<Tensor> {
            Some(Tensor::of_slice(self).to_device(device))
        }
    }
    impl IntoTensorIndex for Vec<i64> {
        fn into_tensor_index(&self, device: Device) -> Option<Tensor> {
            Some(Tensor::of_slice(self.as_slice()).to_device(device))
        }
    }
    impl<const LEN: usize> IntoTensorIndex for [i64; LEN] {
        fn into_tensor_index(&self, device: Device) -> Option<Tensor> {
            Some(Tensor::of_slice(self.as_ref()).to_device(device))
        }
    }
    impl<const LEN: usize> IntoTensorIndex for &[i64; LEN] {
        fn into_tensor_index(&self, device: Device) -> Option<Tensor> {
            Some(Tensor::of_slice(self.as_ref()).to_device(device))
        }
    }

    pub trait IntoIndexList {
        fn into_index_list(self, device: Device) -> Vec<Option<Tensor>>;
    }

    pub const NONE_INDEX: Option<Tensor> = None;

    // slice
    impl<T> IntoIndexList for &[T]
    where
        T: IntoTensorIndex,
    {
        fn into_index_list(self, device: Device) -> Vec<Option<Tensor>> {
            self.iter()
                .map(|index| index.into_tensor_index(device))
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
                .map(|index| index.into_tensor_index(device))
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
                    [ $( <$input_ty as IntoTensorIndex>::into_tensor_index(& $input_arg, device) ),* ]
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

    #[test]
    fn hue_rgb_conv() {
        for _ in 0..1000 {
            {
                let input = Tensor::rand(&[3, 512, 512], (Kind::Float, Device::Cpu));
                let output = input.rgb_to_hsv().hsv_to_rgb();
                assert!(bool::from((input - output).abs().le(1e-10).all()))
            }

            {
                let input = Tensor::rand(&[3, 512, 512], (Kind::Float, Device::Cpu));
                let output = input.hsv_to_rgb().rgb_to_hsv();
                assert!(bool::from((input - output).abs().le(1e-10).all()))
            }
        }
    }
}
