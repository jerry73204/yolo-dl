use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RandomAffineInit {
    pub rotate_radians: Option<R64>,
    pub translation: Option<R64>,
    pub scale: Option<(R64, R64)>,
    pub shear: Option<R64>,
    pub horizontal_flip: bool,
    pub vertical_flip: bool,
}

impl RandomAffineInit {
    pub fn build(self) -> Result<RandomAffine> {
        let Self {
            rotate_radians,
            translation,
            scale,
            shear,
            horizontal_flip,
            vertical_flip,
        } = self;

        let rotate_radians = rotate_radians
            .map(|val| {
                ensure!(val >= 0.0, "rotate_radians must be non-negative");
                Ok(val.raw())
            })
            .transpose()?;
        let translation = translation
            .map(|val| {
                ensure!(val >= 0.0, "translation must be non-negative");
                Ok(val.raw())
            })
            .transpose()?;
        let scale = scale
            .map(|(lo, up)| {
                ensure!(lo >= 0.0, "scale min must be non-negative");
                ensure!(up >= 0.0, "scale max must be non-negative");
                ensure!(lo <= up, "scale min must not exceed scale max");
                Ok((lo.raw(), up.raw()))
            })
            .transpose()?;
        let shear = shear
            .map(|val| {
                ensure!(val >= 0.0, "shear must be non-negative");
                Ok(val.raw())
            })
            .transpose()?;

        Ok(RandomAffine {
            rotate_radians,
            translation,
            scale,
            shear,
            horizontal_flip,
            vertical_flip,
        })
    }
}

impl Default for RandomAffineInit {
    fn default() -> Self {
        Self {
            rotate_radians: None,
            translation: None,
            scale: None,
            shear: None,
            horizontal_flip: false,
            vertical_flip: false,
        }
    }
}

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
    pub fn forward(
        &self,
        orig_image: &Tensor,
        orig_bboxes: &[LabeledRatioBBox],
    ) -> Result<(Tensor, Vec<LabeledRatioBBox>)> {
        tch::no_grad(|| {
            let device = orig_image.device();
            let (channels, height, width) = orig_image.size3()?;

            // sample affine transforms per image
            let transform = {
                let mut rng = StdRng::from_entropy();
                let transform = {
                    let transform = Tensor::eye(3, (Kind::Float, device));
                    let transform = if self.horizontal_flip {
                        if rng.gen::<bool>() {
                            let flip = Tensor::of_slice(
                                &[
                                    [-1.0, 0.0, 0.0], // row 1
                                    [0.0, 1.0, 0.0],  // row 2
                                    [0.0, 0.0, 1.0],  // row 3
                                ]
                                .flat(),
                            )
                            .view([3, 3]);

                            flip.matmul(&transform)
                        } else {
                            transform
                        }
                    } else {
                        transform
                    };
                    let transform = if self.vertical_flip {
                        if rng.gen::<bool>() {
                            let flip = Tensor::of_slice(
                                &[
                                    [1.0, 0.0, 0.0],  // row 1
                                    [0.0, -1.0, 0.0], // row 2
                                    [0.0, 0.0, 1.0],  // row 3
                                ]
                                .flat(),
                            )
                            .view([3, 3]);
                            flip.matmul(&transform)
                        } else {
                            transform
                        }
                    } else {
                        transform
                    };
                    let transform = match self.scale {
                        Some((lower, upper)) => {
                            let ratio = rng.gen_range(lower, upper) as f32;
                            let scaling = Tensor::of_slice(
                                &[
                                    [ratio, 0.0, 0.0], // row 1
                                    [0.0, ratio, 0.0], // row 2
                                    [0.0, 0.0, 1.0],   // row 3
                                ]
                                .flat(),
                            )
                            .view([3, 3]);
                            scaling.matmul(&transform)
                        }
                        None => transform,
                    };
                    let transform = match self.shear {
                        Some(max_shear) => {
                            let shear = rng.gen_range(-max_shear, max_shear) as f32;
                            let translation = Tensor::of_slice(
                                &[
                                    [1.0 + shear, 0.0, 0.0], // row 1
                                    [0.0, 1.0 + shear, 0.0], // row 2
                                    [0.0, 0.0, 1.0],         // row 3
                                ]
                                .flat(),
                            )
                            .view([3, 3]);
                            translation.matmul(&transform)
                        }
                        None => transform,
                    };
                    let transform = match self.rotate_radians {
                        Some(max_randians) => {
                            let angle = rng.gen_range(-max_randians, max_randians);
                            let cos = angle.cos() as f32;
                            let sin = angle.sin() as f32;
                            let rotation = Tensor::of_slice(
                                &[
                                    [cos, -sin, 0.0], // row 1
                                    [sin, cos, 0.0],  // row 2
                                    [0.0, 0.0, 1.0],  // row 3
                                ]
                                .flat(),
                            )
                            .view([3, 3]);
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

                            let translation = Tensor::of_slice(
                                &[
                                    [1.0, 0.0, horizontal_translation], // row 1
                                    [0.0, 1.0, vertical_translation],   // row 2
                                    [0.0, 0.0, 1.0],                    // row 3
                                ]
                                .flat(),
                            )
                            .view([3, 3]);
                            translation.matmul(&transform)
                        }
                        None => transform,
                    };

                    let transform = transform.to_device(device);
                    transform
                };

                transform
            };

            // transform image
            let affine_grid = Tensor::affine_grid_generator(
                &transform.i((0..2, ..)).view([1, 2, 3]), // remove the last row
                &[1, channels, height, width],
                false,
            );
            let new_image = orig_image
                .view([1, channels, height, width])
                .grid_sampler(
                    &affine_grid,
                    // See https://github.com/pytorch/pytorch/blob/f597ac6efc70431e66d945c16fa12b767989b032/aten/src/ATen/native/GridSampler.h#L10-L11
                    0,
                    0,
                    false,
                )
                .view([channels, height, width]);

            // transform bboxes
            let (orig_corners_vec, category_id_vec) = orig_bboxes
                .into_iter()
                .map(|label| {
                    let LabeledPixelBBox {
                        ref bbox,
                        category_id,
                    } = label.to_r64_bbox(height as usize, width as usize);
                    let [orig_t, orig_l, orig_b, orig_r] = bbox.map(|val| val.raw()).tlbr();
                    let orig_corners = Tensor::of_slice(
                        &[
                            [orig_t, orig_l, 1.0],
                            [orig_t, orig_r, 1.0],
                            [orig_b, orig_l, 1.0],
                            [orig_b, orig_r, 1.0],
                        ]
                        .flat(),
                    )
                    .view([4, 3]);

                    (orig_corners, category_id)
                })
                .unzip_n_vec();

            let orig_corners = Tensor::cat(&orig_corners_vec, 0);

            // transform corner points
            let new_corners = transform
                .matmul(&orig_corners.transpose(0, 1))
                .transpose(0, 1);

            // merge with category ids
            let components: Vec<f32> = new_corners.into();
            let new_corners_slice: &[[f32; 3]] = components.as_slice().nest();

            let new_bboxes: Vec<_> = new_corners_slice
                .iter()
                .chunks(4)
                .into_iter()
                .zip_eq(category_id_vec)
                .filter_map(|(points, category_id)| {
                    let points: Vec<_> = points
                        .map(|&[y, x, _]| [R64::new(y as f64), R64::new(x as f64)])
                        .collect();
                    let bbox_t = points
                        .iter()
                        .map(|&[y, _x]| y)
                        .min()
                        .unwrap()
                        .max(R64::new(0.0));
                    let bbox_b = points
                        .iter()
                        .map(|&[y, _x]| y)
                        .max()
                        .unwrap()
                        .min(R64::new(height as f64));
                    let bbox_l = points
                        .iter()
                        .map(|&[_y, x]| x)
                        .min()
                        .unwrap()
                        .max(R64::new(0.0));
                    let bbox_r = points
                        .iter()
                        .map(|&[_y, x]| x)
                        .max()
                        .unwrap()
                        .min(R64::new(width as f64));

                    // remove out of bound bboxes
                    if abs_diff_eq!((bbox_b - bbox_t).raw(), 0.0)
                        || abs_diff_eq!((bbox_r - bbox_l).raw(), 0.0)
                    {
                        return None;
                    }

                    let new_bbox = LabeledPixelBBox {
                        bbox: PixelBBox::try_from_tlbr([bbox_t, bbox_l, bbox_b, bbox_r]).unwrap(),
                        category_id,
                    }
                    .to_ratio_bbox(height as usize, width as usize)
                    .unwrap();

                    Some(new_bbox)
                })
                .collect();

            Ok((new_image, new_bboxes))
        })
    }
}
