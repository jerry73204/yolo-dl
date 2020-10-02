use crate::common::*;

pub trait TensorEx {
    fn fill_rect_(
        &mut self,
        top: i64,
        left: i64,
        bottom: i64,
        right: i64,
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
    ) -> Result<Tensor>;
}

impl TensorEx for Tensor {
    fn fill_rect_(
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
                let expanded_color = color.view([1, n_channels, 1, 1]).expand_as(&rect);
                rect.copy_(&expanded_color);
            }
            &[n_channels, _height, _width] => {
                ensure!(
                    color.size1()? == n_channels,
                    "the number of channels of input and color tensors do not match"
                );
                let mut rect = self.i((.., top..bottom, left..right));
                let expanded_color = color.view([n_channels, 1, 1]).expand_as(&rect);
                rect.copy_(&expanded_color);
            }
            _ => bail!("invalid shape: expect three or four dims"),
        };

        Ok(self.shallow_clone())
    }

    fn draw_rect_(
        &mut self,
        top: i64,
        left: i64,
        bottom: i64,
        right: i64,
        stroke: usize,
        color: &Tensor,
    ) -> Result<Tensor> {
        let (outer_top, outer_left, outer_bottom, outer_right) = {
            let half_stroke = (stroke / 2) as i64;
            (
                top - half_stroke,
                left - half_stroke,
                bottom + half_stroke,
                right + half_stroke,
            )
        };
        let (inner_top, inner_left, inner_bottom, inner_right) = {
            let stroke = stroke as i64;
            (
                outer_top + stroke,
                outer_left + stroke,
                outer_bottom - stroke,
                outer_right - stroke,
            )
        };

        // draw top edge
        let _ = self.fill_rect_(outer_top, outer_left, inner_top, outer_right, color)?;

        // draw left edge
        let _ = self.fill_rect_(outer_top, outer_left, outer_bottom, inner_left, color)?;

        // draw bottom edge
        let _ = self.fill_rect_(inner_bottom, outer_left, outer_bottom, outer_right, color)?;

        // draw right edge
        let _ = self.fill_rect_(outer_top, inner_right, outer_bottom, outer_right, color)?;

        Ok(self.shallow_clone())
    }
}
