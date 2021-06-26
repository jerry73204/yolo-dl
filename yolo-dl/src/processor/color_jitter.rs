//! The random color distortion algorithm.

use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ColorJitterInit {
    pub hue_shift: Option<R64>,
    pub saturation_shift: Option<R64>,
    pub value_shift: Option<R64>,
}

impl ColorJitterInit {
    pub fn build(self) -> ColorJitter {
        let Self {
            hue_shift,
            saturation_shift,
            value_shift,
        } = self;

        ColorJitter {
            max_hue_shift: hue_shift.map(R64::raw),
            max_saturation_shift: saturation_shift.map(R64::raw),
            max_value_shift: value_shift.map(R64::raw),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ColorJitter {
    max_hue_shift: Option<f64>,
    max_saturation_shift: Option<f64>,
    max_value_shift: Option<f64>,
}

impl ColorJitter {
    pub fn forward(&self, rgb: &Tensor) -> Result<Tensor> {
        tch::no_grad(|| -> Result<_> {
            let (channels, _height, _width) = rgb.size3()?;
            ensure!(
                channels == 3,
                "channel size must be 3, but get {}",
                channels
            );

            let mut rng = StdRng::from_entropy();

            let hsv = rgb.rgb_to_hsv();
            let hue = hsv.select(0, 0);
            let saturation = hsv.select(0, 1);
            let value = hsv.select(0, 2);

            if let Some(max_shift) = self.max_hue_shift {
                let shift = rng.gen_range((-max_shift)..max_shift);
                let _ = hue.g_add_scalar(shift + 1.0).fmod_(1.0);
            }

            if let Some(max_shift) = self.max_saturation_shift {
                let shift = rng.gen_range((-max_shift)..max_shift);
                let _ = saturation.g_add_scalar(shift).clamp_(0.0, 1.0);
            }

            if let Some(max_shift) = self.max_value_shift {
                let shift = rng.gen_range((-max_shift)..max_shift);
                let _ = value.g_add_scalar(shift).clamp_(0.0, 1.0);
            }

            let new_rgb = hsv.hsv_to_rgb();

            Ok(new_rgb)
        })
    }
}
