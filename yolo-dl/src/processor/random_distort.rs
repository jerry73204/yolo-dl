use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RandomDistortInit {
    pub hue_shift: Option<R64>,
    pub saturation_scale: Option<R64>,
    pub value_scale: Option<R64>,
}

impl RandomDistortInit {
    pub fn build(self) -> RandomDistort {
        let Self {
            hue_shift,
            saturation_scale,
            value_scale,
        } = self;

        RandomDistort {
            max_hue_shift: hue_shift.map(R64::raw),
            max_saturation_scale: saturation_scale.map(R64::raw),
            max_value_scale: value_scale.map(R64::raw),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RandomDistort {
    max_hue_shift: Option<f64>,
    max_saturation_scale: Option<f64>,
    max_value_scale: Option<f64>,
}

impl RandomDistort {
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
            let mut hue = hsv.select(0, 0);
            let mut saturation = hsv.select(0, 1);
            let mut value = hsv.select(0, 2);

            if let Some(max_shift) = self.max_hue_shift {
                let shift = rng.gen_range(-max_shift, max_shift);
                let _ = hue.g_add_1(shift);
            }

            if let Some(max_scale) = self.max_saturation_scale {
                let scale = rng.gen_range(1.0 / max_scale, max_scale);
                let _ = saturation.g_mul_1(scale);
            }

            if let Some(max_scale) = self.max_value_scale {
                let scale = rng.gen_range(1.0 / max_scale, max_scale);
                let _ = value.g_mul_1(scale);
            }

            let new_rgb = hsv.hsv_to_rgb();

            Ok(new_rgb)
        })
    }
}
