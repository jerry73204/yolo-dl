use crate::common::*;

// CowTensorSlice

pub enum CowTensor<'a> {
    Borrowed(&'a Tensor),
    Owned(Tensor),
}

impl<'a> CowTensor<'a> {
    pub fn into_owned(self) -> Tensor {
        match self {
            Self::Borrowed(borrowed) => borrowed.shallow_clone(),
            Self::Owned(owned) => owned,
        }
    }
}

impl<'a> From<&'a Tensor> for CowTensor<'a> {
    fn from(from: &'a Tensor) -> Self {
        Self::Borrowed(from)
    }
}

impl<'a> From<Tensor> for CowTensor<'a> {
    fn from(from: Tensor) -> Self {
        Self::Owned(from)
    }
}

pub trait TensorEx {
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

    fn crop_by_ratio(&self, top: Ratio, bottom: Ratio, left: Ratio, right: Ratio)
        -> Result<Tensor>;
}

impl TensorEx for Tensor {
    fn f_fill_rect_(
        &mut self,
        top: i64,
        left: i64,
        bottom: i64,
        right: i64,
        color: &Tensor,
    ) -> Result<Tensor> {
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
        };

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
            &[_bsize, n_channels, height, width] => (n_channels, height, width),
            &[n_channels, height, width] => (n_channels, height, width),
            _ => bail!("invalid shape: expect three or four dims"),
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

        // draw t edge
        let _ = self.f_fill_rect_(outer_t, outer_l, inner_t, outer_r, color)?;

        // draw l edge
        let _ = self.f_fill_rect_(outer_t, outer_l, outer_b, inner_l, color)?;

        // draw b edge
        let _ = self.f_fill_rect_(inner_b, outer_l, outer_b, outer_r, color)?;

        // draw r edge
        let _ = self.f_fill_rect_(outer_t, inner_r, outer_b, outer_r, color)?;

        Ok(self.shallow_clone())
    }

    fn f_batch_fill_rect_(&mut self, btlbrs: &[[i64; 5]], color: &Tensor) -> Result<Tensor> {
        let (batch_size, n_channels, height, width) = self.size4()?;
        ensure!(
            n_channels == color.size1()?,
            "number of channels does not match"
        );

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

        let _ = self.f_batch_fill_rect_(&bboxes_t, color)?;
        let _ = self.f_batch_fill_rect_(&bboxes_l, color)?;
        let _ = self.f_batch_fill_rect_(&bboxes_b, color)?;
        let _ = self.f_batch_fill_rect_(&bboxes_r, color)?;

        Ok(self.shallow_clone())
    }

    fn crop_by_ratio(
        &self,
        top: Ratio,
        bottom: Ratio,
        left: Ratio,
        right: Ratio,
    ) -> Result<Tensor> {
        ensure!(left < right, "invalid range");
        ensure!(top < bottom, "invalid range");

        let (_channels, height, width) = self.size3()?;
        let height = height as f64;
        let width = width as f64;

        let top = (f64::from(top) * height) as i64;
        let bottom = (f64::from(bottom) * height) as i64;
        let left = (f64::from(left) * width) as i64;
        let right = (f64::from(right) * width) as i64;

        let cropped = self.i((.., top..bottom, left..right));

        Ok(cropped)
    }
}
