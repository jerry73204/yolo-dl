use crate::common::*;

#[derive(Debug, Clone)]
pub struct UpSample2D {
    scale: f64,
}

impl UpSample2D {
    pub fn new(scale: f64) -> Result<Self> {
        ensure!(
            scale.is_finite() && scale.is_sign_positive(),
            "invalid scale value"
        );
        Ok(Self { scale })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let Self { scale } = *self;
        let (_b, _c, in_h, in_w) = input.size4()?;
        let out_h = (in_h as f64 * scale) as i64;
        let out_w = (in_w as f64 * scale) as i64;
        let output = input.upsample_nearest2d(&[out_h, out_w], None, None);
        Ok(output)
    }
}
