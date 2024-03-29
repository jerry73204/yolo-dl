//! The random affine transformation algorithm.

use crate::{common::*, label::RatioLabel};
use bbox::{prelude::*, CyCxHW};
use label::Label;
use tch_goodies::Ratio;

/// Random affine transformation processor initializer.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct RandomAffineInit {
    /// The probability to apply rotation.
    pub rotate_prob: Option<R64>,
    /// The maximum rotation angle in radians.
    pub rotate_radians: Option<R64>,
    /// The probability to translate image location.
    pub translation_prob: Option<R64>,
    /// The maximum translation distance along X and Y axises in ratio unit.
    pub translation: Option<R64>,
    /// The probability to scale up or down the image size.
    pub scale_prob: Option<R64>,
    /// The pair of minimum and maximum scaling ratio.
    pub scale: Option<(R64, R64)>,
    // pub shear_prob: Option<Ratio>,
    // pub shear: Option<R64>,
    /// The probability to apply horizontal flip.
    pub horizontal_flip_prob: Option<R64>,
    /// The probability to apply vertical flip.
    pub vertical_flip_prob: Option<R64>,
    pub min_bbox_size: Option<R64>,
    pub min_bbox_cropping_ratio: Option<R64>,
}

impl RandomAffineInit {
    pub fn build(self) -> Result<RandomAffine> {
        let Self {
            rotate_prob,
            rotate_radians,
            translation_prob,
            translation,
            scale_prob,
            scale,
            // shear_prob,
            // shear,
            horizontal_flip_prob,
            vertical_flip_prob,
            min_bbox_size,
            min_bbox_cropping_ratio,
        } = self;

        ensure!(
            matches!(rotate_radians, Some(val) if val >= 0.0),
            "rotate_radians must be non-negative"
        );

        ensure!(
            matches!(translation, Some(val) if val >= 0.0),
            "translation must be non-negative"
        );

        ensure!(
            matches!(scale, Some((lo, up)) if lo >= 0.0 && up >= 0.0 && lo <= up),
            "scale min and max must be non-negative, and min must be lower or equal to max"
        );

        ensure!(
            matches!(min_bbox_size, Some(size) if (0.0..=1.0).contains(&size.raw())),
            "min_bbox_size must be between 0.0 and 1.0"
        );

        ensure!(
            matches!(min_bbox_cropping_ratio, Some(ratio) if (0.0..=1.0).contains(&ratio.raw())),
            "min_bbox_cropping_ratio must be between 0.0 and 1.0"
        );

        Ok(RandomAffine {
            rotate_prob,
            rotate_radians,
            translation_prob,
            translation,
            scale_prob,
            scale,
            // shear_prob: shear_prob.as_ref().map(Ratio::to_f64),
            // shear,
            horizontal_flip_prob,
            vertical_flip_prob,
            min_bbox_size,
            min_bbox_cropping_ratio,
        })
    }
}

/// Random affine transformation processor.
#[derive(Debug, Clone)]
pub struct RandomAffine {
    rotate_prob: Option<R64>,
    rotate_radians: Option<R64>,
    translation_prob: Option<R64>,
    translation: Option<R64>,
    scale_prob: Option<R64>,
    scale: Option<(R64, R64)>,
    // shear_prob: Option<R64>,
    // shear: Option<R64>,
    horizontal_flip_prob: Option<R64>,
    vertical_flip_prob: Option<R64>,
    min_bbox_size: Option<R64>,
    min_bbox_cropping_ratio: Option<R64>,
}

impl RandomAffine {
    /// Apply random affine transformation on an image and boxes.
    pub fn forward(
        &self,
        orig_image: &Tensor,
        orig_bboxes: &[RatioLabel],
    ) -> Result<(Tensor, Vec<RatioLabel>)> {
        tch::no_grad(|| {
            let device = orig_image.device();
            let (channels, height, width) = orig_image.size3()?;

            // sample affine transforms per image
            // the zero point of input space of affine matrix is at image center,
            // and image height and width takes 2 units
            let transform = {
                let mut rng = StdRng::from_entropy();

                let transform = Tensor::eye(3, (Kind::Float, Device::Cpu));

                let transform = match self.horizontal_flip_prob {
                    Some(prob) => {
                        if rng.gen_bool(prob.raw()) {
                            let flip = Tensor::of_slice(
                                [
                                    [-1f32, 0.0, 0.0], // row 1
                                    [0.0, 1.0, 0.0],   // row 2
                                    [0.0, 0.0, 1.0],   // row 3
                                ]
                                .flat(),
                            )
                            .view([3, 3]);

                            flip.matmul(&transform)
                        } else {
                            transform
                        }
                    }
                    None => transform,
                };

                let transform = match self.vertical_flip_prob {
                    Some(prob) => {
                        if rng.gen_bool(prob.raw()) {
                            let flip = Tensor::of_slice(
                                [
                                    [1f32, 0.0, 0.0], // row 1
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
                    }
                    None => transform,
                };

                let transform = match (self.scale_prob, self.scale) {
                    (Some(prob), Some((lower, upper))) => {
                        if rng.gen_bool(prob.raw()) {
                            let ratio = rng.gen_range(lower.raw()..upper.raw()) as f32;
                            let scaling = Tensor::of_slice(
                                [
                                    [ratio, 0.0, 0.0], // row 1
                                    [0.0, ratio, 0.0], // row 2
                                    [0.0, 0.0, 1.0],   // row 3
                                ]
                                .flat(),
                            )
                            .view([3, 3]);
                            scaling.matmul(&transform)
                        } else {
                            transform
                        }
                    }
                    _ => transform,
                };

                // let transform = match (self.shear_prob, self.shear) {
                //     (Some(prob), Some(max_shear)) => {
                //         if rng.gen_bool(prob) {
                //             let shear = rng.gen_range((-max_shear)..max_shear) as f32;
                //             let shear = Tensor::of_slice(
                //                 &[
                //                     [1.0 + shear, 0.0, 0.0], // row 1
                //                     [0.0, 1.0 + shear, 0.0], // row 2
                //                     [0.0, 0.0, 1.0],         // row 3
                //                 ]
                //                 .flat(),
                //             )
                //             .view([3, 3]);
                //             shear.matmul(&transform)
                //         } else {
                //             transform
                //         }
                //     }
                //     _ => transform,
                // };

                let transform = match (self.rotate_prob, self.rotate_radians) {
                    (Some(prob), Some(max_randians)) => {
                        if rng.gen_bool(prob.raw()) {
                            let angle = rng.gen_range((-max_randians.raw())..=max_randians.raw());
                            let cos = angle.cos() as f32;
                            let sin = angle.sin() as f32;
                            let rotation = Tensor::of_slice(
                                [
                                    [cos, -sin, 0.0], // row 1
                                    [sin, cos, 0.0],  // row 2
                                    [0.0, 0.0, 1.0],  // row 3
                                ]
                                .flat(),
                            )
                            .view([3, 3]);
                            rotation.matmul(&transform)
                        } else {
                            transform
                        }
                    }
                    _ => transform,
                };

                let transform = match (self.translation_prob, self.translation) {
                    (Some(prob), Some(max_translation)) => {
                        if rng.gen_bool(prob.raw()) {
                            // whole image height/width takes 2 units, so translations are doubled
                            let horizontal_translation =
                                (rng.gen_range((-max_translation.raw())..max_translation.raw())
                                    * 2.0) as f32;
                            let vertical_translation =
                                (rng.gen_range((-max_translation.raw())..max_translation.raw())
                                    * 2.0) as f32;

                            let translation = Tensor::of_slice(
                                [
                                    [1.0, 0.0, horizontal_translation], // row 1
                                    [0.0, 1.0, vertical_translation],   // row 2
                                    [0.0, 0.0, 1.0],                    // row 3
                                ]
                                .flat(),
                            )
                            .view([3, 3]);
                            translation.matmul(&transform)
                        } else {
                            transform
                        }
                    }
                    _ => transform,
                };

                transform
            };

            // the inverse() runs on CPU due to (maybe) unreported memory leaking issue on GPU
            let inv_transform = transform.inverse();

            // transform image
            // affine_grid_generator() maps from output coordinates to input coordinates,
            // hence we pass the inverse matrix here
            let affine_grid = Tensor::affine_grid_generator(
                &inv_transform.i((..2, ..)).view([1, 2, 3]), // add batch dimension
                &[1, channels, height, width],
                false,
            );
            let new_image = orig_image
                .view([1, channels, height, width])
                .grid_sampler(
                    &affine_grid.to_device(device),
                    // See https://github.com/pytorch/pytorch/blob/f597ac6efc70431e66d945c16fa12b767989b032/aten/src/ATen/native/GridSampler.h#L10-L11
                    0,
                    0,
                    false,
                )
                .view([channels, height, width]);

            // transform bboxes
            let new_bboxes = if !orig_bboxes.is_empty() {
                let (orig_corners_vec, class_vec) = orig_bboxes
                    .iter()
                    .map(|label| {
                        // let Label {
                        //     rect: ref cycxhw,
                        //     class,
                        // } = **label;
                        // let tlbr: TLBR<_> = cycxhw.cast::<f32>().into();
                        let rect = label.rect.clone().cast::<f32>();
                        let orig_t = rect.t();
                        let orig_l = rect.l();
                        let orig_b = rect.b();
                        let orig_r = rect.r();

                        let orig_corners = Tensor::of_slice(
                            [
                                [orig_l, orig_t, 1.0],
                                [orig_l, orig_b, 1.0],
                                [orig_r, orig_t, 1.0],
                                [orig_r, orig_b, 1.0],
                            ]
                            .flat(),
                        )
                        .view([4, 3]);

                        (orig_corners, label.class)
                    })
                    .unzip_n_vec();

                let orig_corners = Tensor::cat(&orig_corners_vec, 0);

                // transform corner points, it runs on CPU
                let new_corners = {
                    let coord_change = Tensor::of_slice(
                        [
                            [2f32, 0.0, -1.0], // row 1
                            [0.0, 2.0, -1.0],  // row 2
                            [0.0, 0.0, 1.0],   // row 3
                        ]
                        .flat(),
                    )
                    .view([3, 3])
                    .transpose(0, 1);

                    // move center of image to zero position and transpose
                    let input = orig_corners.matmul(&coord_change);

                    // transform coordinates
                    let output = input.matmul(&transform.transpose(0, 1));

                    // move zero position to center of image and transpose
                    output.matmul(&coord_change.inverse())
                };

                // merge with category ids
                let components: Vec<f32> = new_corners.into();
                let new_corners_slice: &[[f32; 3]] = components.as_slice().nest();

                let new_bboxes: Vec<_> = new_corners_slice
                    .iter()
                    .chunks(4)
                    .into_iter()
                    .zip_eq(class_vec)
                    .zip_eq(orig_bboxes)
                    .filter_map(|((points, class), orig_bbox)| {
                        let points: Vec<_> = points
                            .map(|&[x, y, _]| [R64::new(y as f64), R64::new(x as f64)])
                            .collect();
                        let bbox_t = points.iter().map(|&[y, _x]| y).min().unwrap();
                        let bbox_b = points.iter().map(|&[y, _x]| y).max().unwrap();
                        let bbox_l = points.iter().map(|&[_y, x]| x).min().unwrap();
                        let bbox_r = points.iter().map(|&[_y, x]| x).max().unwrap();

                        // remove out of bound bboxes
                        if bbox_t >= 1.0 || bbox_b <= 0.0 || bbox_l >= 1.0 || bbox_r <= 0.0 {
                            return None;
                        }

                        let bbox_t = bbox_t.max(r64(0.0));
                        let bbox_l = bbox_l.max(r64(0.0));
                        let bbox_b = bbox_b.min(r64(1.0));
                        let bbox_r = bbox_r.min(r64(1.0));

                        // remove small bboxes
                        let h = bbox_b - bbox_t;
                        let w = bbox_r - bbox_l;

                        if let Some(min_bbox_size) = self.min_bbox_size {
                            if h < min_bbox_size || w < min_bbox_size {
                                return None;
                            }
                        }

                        if let Some(min_bbox_cropping_ratio) = self.min_bbox_cropping_ratio {
                            let orig_area = orig_bbox.rect.area();
                            let new_area = h * w;
                            if new_area < orig_area * min_bbox_cropping_ratio {
                                return None;
                            }
                        }

                        // construct output bbox
                        let new_bbox: CyCxHW<R64> =
                            CyCxHW::from_tlbr([bbox_t, bbox_l, bbox_b, bbox_r]).cast::<R64>();

                        let new_label = Ratio(Label {
                            rect: new_bbox,
                            class,
                        });

                        Some(new_label)
                    })
                    .collect();

                new_bboxes
            } else {
                vec![]
            };

            Ok((new_image, new_bboxes))
        })
    }
}
