use crate::common::*;

// random affine

#[derive(Debug, Clone)]
pub struct RandomAffine {
    rotate_radians: Option<f64>,
    translation: Option<f64>,
    scale: Option<(f64, f64)>,
    shear: Option<f64>,
    horizontal_flip: bool,
    vertical_flip: bool,
}

impl RandomAffine {
    pub fn new(
        rotate_degrees: impl Into<Option<R64>>,
        translation: impl Into<Option<R64>>,
        scale: impl Into<Option<(R64, R64)>>,
        shear: impl Into<Option<R64>>,
        horizontal_flip: impl Into<Option<bool>>,
        vertical_flip: impl Into<Option<bool>>,
    ) -> Self {
        let rotate_radians = rotate_degrees
            .into()
            .map(|degrees| degrees.raw().to_radians());
        let translation = translation.into().map(|val| val.raw());
        let scale = scale.into().map(|(lo, up)| (lo.raw(), up.raw()));
        let shear = shear.into().map(|val| val.raw());
        let horizontal_flip = horizontal_flip.into().unwrap_or(true);
        let vertical_flip = vertical_flip.into().unwrap_or(false);

        if let Some(val) = rotate_radians {
            assert!(val >= 0.0);
        }

        if let Some(val) = translation {
            assert!(val >= 0.0);
        }

        if let Some((min, max)) = scale {
            assert!(min >= 0.0);
            assert!(min >= 0.0);
            assert!(min <= max);
        }

        if let Some(val) = shear {
            assert!(val >= 0.0);
        }

        Self {
            rotate_radians,
            translation,
            scale,
            shear,
            horizontal_flip,
            vertical_flip,
        }
    }

    pub fn batch_random_affine(&self, image: &Tensor) -> Tensor {
        let (bsize, channels, height, width) = image.size4().unwrap();
        let device = image.device();
        let mut rng = StdRng::from_entropy();

        let affine_transforms: Vec<_> = (0..bsize)
            .map(|_| {
                let transform = Tensor::eye(3, (Kind::Float, device));
                let transform = match self.horizontal_flip {
                    true => {
                        if rng.gen::<bool>() {
                            let flip: Tensor = Array2::<f32>::from_shape_vec(
                                (3, 3),
                                vec![
                                    -1.0, 0.0, 0.0, // row 1
                                    0.0, 1.0, 0.0, // row 2
                                    0.0, 0.0, 1.0, // row 3
                                ],
                            )
                            .unwrap()
                            .try_into()
                            .unwrap();

                            flip.matmul(&transform)
                        } else {
                            transform
                        }
                    }
                    false => transform,
                };
                let transform = match self.vertical_flip {
                    true => {
                        if rng.gen::<bool>() {
                            let flip = Tensor::try_from(
                                Array2::<f32>::from_shape_vec(
                                    (3, 3),
                                    vec![
                                        1.0, 0.0, 0.0, // row 1
                                        0.0, -1.0, 0.0, // row 2
                                        0.0, 0.0, 1.0, // row 3
                                    ],
                                )
                                .unwrap(),
                            )
                            .unwrap();

                            flip.matmul(&transform)
                        } else {
                            transform
                        }
                    }
                    false => transform,
                };
                let transform = match self.scale {
                    Some((lower, upper)) => {
                        let ratio = rng.gen_range(lower, upper) as f32;

                        let scaling = Tensor::try_from(
                            Array2::from_shape_vec(
                                (3, 3),
                                vec![
                                    ratio, 0.0, 0.0, // row 1
                                    0.0, ratio, 0.0, // row 2
                                    0.0, 0.0, 1.0, // row 3
                                ],
                            )
                            .unwrap(),
                        )
                        .unwrap();

                        scaling.matmul(&transform)
                    }
                    None => transform,
                };
                let transform = match self.shear {
                    Some(max_shear) => {
                        let shear = rng.gen_range(-max_shear, max_shear) as f32;

                        let translation = Tensor::try_from(
                            Array2::from_shape_vec(
                                (3, 3),
                                vec![
                                    1.0 + shear,
                                    0.0,
                                    0.0, // row 1
                                    0.0,
                                    1.0 + shear,
                                    0.0, // row 2
                                    0.0,
                                    0.0,
                                    1.0, // row 3
                                ],
                            )
                            .unwrap(),
                        )
                        .unwrap();

                        translation.matmul(&transform)
                    }
                    None => transform,
                };
                let transform = match self.rotate_radians {
                    Some(max_randians) => {
                        let angle = rng.gen_range(-max_randians, max_randians);
                        let cos = angle.cos() as f32;
                        let sin = angle.sin() as f32;

                        let rotation = Tensor::try_from(
                            Array2::from_shape_vec(
                                (3, 3),
                                vec![
                                    cos, -sin, 0.0, // row 1
                                    sin, cos, 0.0, // row 2
                                    0.0, 0.0, 1.0, // row 3
                                ],
                            )
                            .unwrap(),
                        )
                        .unwrap();

                        rotation.matmul(&transform)
                    }
                    None => transform,
                };
                let transform = match self.translation {
                    Some(max_translation) => {
                        let horizontal_translation =
                            (rng.gen_range(-max_translation, max_translation) * height as f64)
                                as f32;
                        let vertical_translation =
                            (rng.gen_range(-max_translation, max_translation) * width as f64)
                                as f32;

                        let translation = Tensor::try_from(
                            Array2::from_shape_vec(
                                (3, 3),
                                vec![
                                    1.0,
                                    0.0,
                                    horizontal_translation, // row 1
                                    0.0,
                                    1.0,
                                    vertical_translation, // row 2
                                    0.0,
                                    0.0,
                                    1.0, // row 3
                                ],
                            )
                            .unwrap(),
                        )
                        .unwrap();

                        translation.matmul(&transform)
                    }
                    None => transform,
                };

                let transform = transform.to_device(device);
                transform
            })
            .collect();

        let batch_affine_transform = Tensor::stack(&affine_transforms, 0);
        let affine_grid = Tensor::affine_grid_generator(
            &batch_affine_transform.i((.., 0..2, ..)), // remove the last row
            &[bsize, channels, height, width],
            false,
        );

        let sampled = image.grid_sampler(
            &affine_grid,
            // See https://github.com/pytorch/pytorch/blob/f597ac6efc70431e66d945c16fa12b767989b032/aten/src/ATen/native/GridSampler.h#L10-L11
            0,
            0,
            false,
        );

        sampled
    }
}
