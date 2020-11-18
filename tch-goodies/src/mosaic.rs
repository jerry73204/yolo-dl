use crate::{bbox::LabeledRatioBBox, common::*, tensor::TensorExt};

#[derive(Debug, Clone)]
pub struct MosaicProcessorInit {
    pub mosaic_margin: f64,
}

impl MosaicProcessorInit {
    pub fn build(self) -> Result<MosaicProcessor> {
        let Self { mosaic_margin } = self;
        ensure!(
            (0.0..0.5).contains(&mosaic_margin),
            "mosaic_margin must be in range 0.0..0.5"
        );

        Ok(MosaicProcessor {
            mosaic_margin,
            rng: StdRng::from_entropy(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct MosaicProcessor {
    mosaic_margin: f64,
    rng: StdRng,
}

impl MosaicProcessor {
    pub fn forward<PairIntoIter, BBoxIntoIter>(
        &mut self,
        input: PairIntoIter,
    ) -> Result<(Tensor, Vec<LabeledRatioBBox>)>
    where
        PairIntoIter: IntoIterator<Item = (Tensor, BBoxIntoIter)>,
        BBoxIntoIter: IntoIterator<Item = LabeledRatioBBox>,
        PairIntoIter::IntoIter: ExactSizeIterator,
    {
        let input_iter = input.into_iter();
        let Self {
            mosaic_margin,
            ref mut rng,
        } = *self;

        ensure!(input_iter.len() == 4, "expect exactly 4 images");

        // random select pivot point

        let ranges = {
            let pivot_row = rng.gen_range(mosaic_margin, 1.0 - mosaic_margin);
            let pivot_col = rng.gen_range(mosaic_margin, 1.0 - mosaic_margin);
            vec![
                [0.0, pivot_row, 0.0, pivot_col],
                [0.0, pivot_row, pivot_col, 1.0],
                [pivot_row, 1.0, 0.0, pivot_col],
                [pivot_row, 1.0, pivot_col, 1.0],
            ]
        };

        // crop images
        let mut crop_iter =
            input_iter
                .zip_eq(ranges.into_iter())
                .scan(None, |expect_shape, args| {
                    let result = || -> Result<_> {
                        let ((image, bboxes), [top, bottom, left, right]) = args;

                        // ensure image is 3 dimensional
                        let shape = image
                            .size3()
                            .with_context(|| "image must have shape [channels, height, width]")?;

                        // check if every image have identical shape
                        match expect_shape.as_ref() {
                            Some(expect_shape) => {
                                ensure!(*expect_shape == shape, "images must have identical shape")
                            }
                            None => *expect_shape = Some(shape),
                        }

                        // crop image
                        let cropped_image =
                            tch::no_grad(|| image.f_crop_by_ratio(top, left, bottom, right))?;

                        // crop bbox
                        let cropped_bboxes = bboxes
                            .into_iter()
                            .filter_map(|bbox| {
                                bbox.crop(top.into(), bottom.into(), left.into(), right.into())
                            })
                            .collect_vec();

                        Ok((cropped_image, cropped_bboxes))
                    }();

                    Some(result)
                });

        let (cropped_tl, bboxes_tl) = crop_iter.next().unwrap()?;
        let (cropped_tr, bboxes_tr) = crop_iter.next().unwrap()?;
        let (cropped_bl, bboxes_bl) = crop_iter.next().unwrap()?;
        let (cropped_br, bboxes_br) = crop_iter.next().unwrap()?;
        debug_assert!(crop_iter.next().is_none());

        // merge cropped images
        let (merged_image, merged_bboxes) = tch::no_grad(|| {
            let merged_top = Tensor::cat(&[cropped_tl, cropped_tr], 2);
            let merged_bottom = Tensor::cat(&[cropped_bl, cropped_br], 2);
            let merged_image = Tensor::cat(&[merged_top, merged_bottom], 1);
            // debug_assert_eq!(merged_image.size3().unwrap(), (3, image_size, image_size));

            // merge cropped bboxes
            let merged_bboxes: Vec<_> = bboxes_tl
                .into_iter()
                .chain(bboxes_tr.into_iter())
                .chain(bboxes_bl.into_iter())
                .chain(bboxes_br.into_iter())
                .collect();

            (merged_image, merged_bboxes)
        });

        Ok((merged_image, merged_bboxes))
    }
}
