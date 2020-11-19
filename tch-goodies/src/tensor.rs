use crate::common::*;

pub trait TensorExt {
    fn f_fill_rect_(
        &mut self,
        top: i64,
        left: i64,
        bottom: i64,
        right: i64,
        color: &Tensor,
    ) -> Result<Tensor>;
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
    fn f_draw_rect_(
        &mut self,
        top: i64,
        left: i64,
        bottom: i64,
        right: i64,
        stroke: usize,
        color: &Tensor,
    ) -> Result<Tensor>;
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
    fn f_batch_fill_rect_(&mut self, btlbrs: &[[i64; 5]], color: &Tensor) -> Result<Tensor>;
    fn batch_fill_rect_(&mut self, btlbrs: &[[i64; 5]], color: &Tensor) -> Tensor {
        self.f_batch_fill_rect_(btlbrs, color).unwrap()
    }
    fn f_batch_draw_rect_(
        &mut self,
        btlbrs: &[[i64; 5]],
        stroke: usize,
        color: &Tensor,
    ) -> Result<Tensor>;
    fn batch_draw_rect_(&mut self, btlbrs: &[[i64; 5]], stroke: usize, color: &Tensor) -> Tensor {
        self.f_batch_draw_rect_(btlbrs, stroke, color).unwrap()
    }
    fn f_crop_by_ratio(&self, top: f64, left: f64, bottom: f64, right: f64) -> Result<Tensor>;
    fn crop_by_ratio(&self, top: f64, left: f64, bottom: f64, right: f64) -> Tensor {
        self.f_crop_by_ratio(top, left, bottom, right).unwrap()
    }
    fn resize2d(&self, new_height: i64, new_width: i64) -> Result<Tensor>;
    fn resize2d_exact(&self, new_height: i64, new_width: i64) -> Result<Tensor>;
    fn resize2d_letterbox(&self, new_height: i64, new_width: i64) -> Result<Tensor>;
    fn swish(&self) -> Tensor;
    fn hard_swish(&self) -> Tensor;
    fn mish(&self) -> Tensor;
    fn hard_mish(&self) -> Tensor;
    // fn normalize_channels(&self) -> Tensor;
    // fn normalize_channels_softmax(&self) -> Tensor;
}

impl TensorExt for Tensor {
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
                    let expanded_color = color.f_view([1, n_channels, 1, 1])?.f_expand_as(&rect)?;
                    rect.f_copy_(&expanded_color)?;
                }
                &[n_channels, _height, _width] => {
                    ensure!(
                        color.size1()? == n_channels,
                        "the number of channels of input and color tensors do not match"
                    );
                    let mut rect = self.i((.., top..bottom, left..right));
                    let expanded_color = color.f_view([n_channels, 1, 1])?.f_expand_as(&rect)?;
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
                        let resized =
                            vision::image::resize(&self.select(0, index), new_width, new_height)?;
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
                let inner =
                    vision::image::resize(&(self * 255.0).to_kind(Kind::Uint8), inner_w, inner_h)?
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
                    .map(|index| vision::image::resize(&scaled.select(0, index), inner_w, inner_h))
                    .try_collect()?;
                let inner = Tensor::stack(resized_vec.as_slice(), 0).to_kind(Kind::Float) / 255.0;
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
}

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