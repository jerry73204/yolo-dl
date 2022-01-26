//! The Mosaic mixing algorithm.

use crate::common::*;
use tch_goodies::{RatioCyCxHW, RatioRectLabel, Rect as _, TensorExt};
use tracing::instrument;

/// Mosaic processor initializer.
#[derive(Debug, Clone)]
pub struct MosaicProcessorInit {
    /// The distance from pivot point to image boundary in ratio unit.
    pub mosaic_margin: f64,
    pub min_bbox_size: Option<f64>,
    pub min_bbox_cropping_ratio: Option<f64>,
}

impl MosaicProcessorInit {
    pub fn build(self) -> Result<MosaicProcessor> {
        let Self {
            mosaic_margin,
            min_bbox_size,
            min_bbox_cropping_ratio,
        } = self;
        ensure!(
            (0.0..=0.5).contains(&mosaic_margin),
            "mosaic_margin must be in range 0.0..0.5"
        );

        if let Some(min_bbox_size) = min_bbox_size {
            ensure!(
                (0.0..=1.0).contains(&min_bbox_size),
                "min_bbox_size must be between 0.0 and 1.0"
            );
        }

        if let Some(min_bbox_cropping_ratio) = min_bbox_cropping_ratio {
            ensure!(
                (0.0..=1.0).contains(&min_bbox_cropping_ratio),
                "min_bbox_cropping_ratio must be between 0.0 and 1.0"
            );
        }

        Ok(MosaicProcessor {
            mosaic_margin,
            min_bbox_size,
            min_bbox_cropping_ratio,
        })
    }
}

/// Mosaic processor.
#[derive(Debug, Clone)]
pub struct MosaicProcessor {
    mosaic_margin: f64,
    min_bbox_size: Option<f64>,
    min_bbox_cropping_ratio: Option<f64>,
}

impl MosaicProcessor {
    /// Apply mosaic mixup on a set of 4 images and boxes.
    pub fn forward<PairIntoIter, CyCxHWIntoIter>(
        &self,
        input: PairIntoIter,
    ) -> Result<(Tensor, Vec<RatioRectLabel<R64>>)>
    where
        PairIntoIter: IntoIterator<Item = (Tensor, CyCxHWIntoIter)>,
        CyCxHWIntoIter: IntoIterator<Item = RatioRectLabel<R64>>,
        PairIntoIter::IntoIter: ExactSizeIterator,
    {
        tch::no_grad(|| {
            let input_iter = input.into_iter();
            ensure!(input_iter.len() == 4, "expect exactly 4 images");
            let Self {
                mosaic_margin,
                min_bbox_size,
                min_bbox_cropping_ratio,
            } = *self;
            let mut rng = StdRng::from_entropy();

            // select pivot point randomly and compute margins per image
            let ranges = {
                let pivot_row = rng.gen_range(mosaic_margin..=(1.0 - mosaic_margin));
                let pivot_col = rng.gen_range(mosaic_margin..=(1.0 - mosaic_margin));
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
                            let ((image, bboxes), [margin_t, margin_b, margin_l, margin_r]) = args;

                            // ensure image is 3 dimensional
                            let shape = image.size3().with_context(|| {
                                "image must have shape [channels, height, width]"
                            })?;

                            // check if every image have identical shape
                            match expect_shape.as_ref() {
                                Some(expect_shape) => ensure!(
                                    *expect_shape == shape,
                                    "images must have identical shape"
                                ),
                                None => *expect_shape = Some(shape),
                            }

                            crop_image_bboxes(
                                &image,
                                bboxes,
                                [margin_t, margin_b, margin_l, margin_r],
                                min_bbox_size,
                                min_bbox_cropping_ratio,
                            )
                        }();

                        Some(result)
                    });

            let (cropped_tl, bboxes_tl) = crop_iter.next().unwrap()?;
            let (cropped_tr, bboxes_tr) = crop_iter.next().unwrap()?;
            let (cropped_bl, bboxes_bl) = crop_iter.next().unwrap()?;
            let (cropped_br, bboxes_br) = crop_iter.next().unwrap()?;
            debug_assert!(crop_iter.next().is_none());

            // merge cropped images
            let (merged_image, merged_bboxes) = {
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
            };

            Ok((merged_image, merged_bboxes))
        })
    }
}

/// Multi-threaded mosaic processor initializer.
#[derive(Debug, Clone)]
pub struct ParallelMosaicProcessorInit {
    pub mosaic_margin: f64,
    pub max_workers: Option<usize>,
    pub min_bbox_size: Option<f64>,
    pub min_bbox_cropping_ratio: Option<f64>,
}

impl ParallelMosaicProcessorInit {
    pub fn build(self) -> Result<ParallelMosaicProcessor> {
        let Self {
            mosaic_margin,
            max_workers,
            min_bbox_size,
            min_bbox_cropping_ratio,
        } = self;

        ensure!(
            (0.0..=0.5).contains(&mosaic_margin),
            "mosaic_margin must be between 0.0 and 0.5"
        );

        if let Some(min_bbox_size) = min_bbox_size {
            ensure!(
                (0.0..=1.0).contains(&min_bbox_size),
                "min_bbox_size must be between 0.0 and 1.0"
            );
        }

        if let Some(min_bbox_cropping_ratio) = min_bbox_cropping_ratio {
            ensure!(
                (0.0..=1.0).contains(&min_bbox_cropping_ratio),
                "min_bbox_cropping_ratio must be between 0.0 and 1.0"
            );
        }

        Ok(ParallelMosaicProcessor {
            mosaic_margin,
            max_workers,
            min_bbox_size,
            min_bbox_cropping_ratio,
        })
    }
}

/// Multi-threaded mosaic processor.
#[derive(Debug, Clone)]
pub struct ParallelMosaicProcessor {
    mosaic_margin: f64,
    max_workers: Option<usize>,
    min_bbox_size: Option<f64>,
    min_bbox_cropping_ratio: Option<f64>,
}

impl ParallelMosaicProcessor {
    /// Apply mosaic mixup on a set of 4 images and boxes.
    #[instrument(skip(input))]
    pub async fn forward<PairIntoIter, CyCxHWIntoIter>(
        &self,
        input: PairIntoIter,
    ) -> Result<(Tensor, Vec<RatioRectLabel<R64>>)>
    where
        PairIntoIter: IntoIterator<Item = (Tensor, CyCxHWIntoIter)>,
        CyCxHWIntoIter: 'static + IntoIterator<Item = RatioRectLabel<R64>> + Send,
        PairIntoIter::IntoIter: ExactSizeIterator,
    {
        let pairs: Vec<_> = input.into_iter().collect();
        ensure!(pairs.len() == 4, "expect exactly 4 images");
        let Self {
            mosaic_margin,
            max_workers,
            min_bbox_size,
            min_bbox_cropping_ratio,
        } = *self;
        let mut rng = StdRng::from_entropy();

        // random select pivot point
        let ranges = {
            let pivot_row = rng.gen_range(mosaic_margin..=(1.0 - mosaic_margin));
            let pivot_col = rng.gen_range(mosaic_margin..=(1.0 - mosaic_margin));
            vec![
                [0.0, pivot_row, 0.0, pivot_col],
                [0.0, pivot_row, pivot_col, 1.0],
                [pivot_row, 1.0, 0.0, pivot_col],
                [pivot_row, 1.0, pivot_col, 1.0],
            ]
        };

        // verify image shape
        {
            let shapes: HashSet<_> = pairs
                .iter()
                .map(|(image, _bboxes)| {
                    image
                        .size3()
                        .with_context(|| "image must have shape [channels, height, width]")
                })
                .try_collect()?;
            ensure!(shapes.len() == 1, "input images must have identical shapes");
        }

        // crop images
        let max_workers = max_workers.unwrap_or_else(num_cpus::get);

        let mut crop_iter = stream::iter(pairs.into_iter().zip_eq(ranges.into_iter())).par_map(
            max_workers,
            move |args| {
                let ((image, bboxes), [margin_t, margin_b, margin_l, margin_r]) = args;
                move || -> Result<_> {
                    crop_image_bboxes(
                        &image,
                        bboxes,
                        [margin_t, margin_b, margin_l, margin_r],
                        min_bbox_size,
                        min_bbox_cropping_ratio,
                    )
                }
            },
        );

        let (cropped_tl, bboxes_tl) = crop_iter.next().await.unwrap()?;
        let (cropped_tr, bboxes_tr) = crop_iter.next().await.unwrap()?;
        let (cropped_bl, bboxes_bl) = crop_iter.next().await.unwrap()?;
        let (cropped_br, bboxes_br) = crop_iter.next().await.unwrap()?;
        debug_assert!(crop_iter.next().await.is_none());

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

fn crop_image_bboxes(
    image: &Tensor,
    bboxes: impl IntoIterator<Item = RatioRectLabel<R64>>,
    tlbr: [f64; 4],
    min_bbox_size: Option<f64>,
    min_bbox_cropping_ratio: Option<f64>,
) -> Result<(Tensor, Vec<RatioRectLabel<R64>>)> {
    tch::no_grad(|| {
        let [margin_t, margin_b, margin_l, margin_r] = tlbr;

        // crop image
        let cropped_image = image.f_crop_by_ratio(margin_t, margin_l, margin_b, margin_r)?;

        // crop bbox
        let cropped_bboxes = bboxes
            .into_iter()
            .filter_map(|bbox| {
                let roi = RatioCyCxHW::from_tlbr(margin_t, margin_l, margin_b, margin_r)
                    .unwrap()
                    .cast::<R64>()
                    .unwrap();
                let cropped = bbox.intersect_with(&roi)?;

                let check1 = min_bbox_size
                    .map(|min_bbox_size| {
                        cropped.h() >= min_bbox_size && cropped.w() >= min_bbox_size
                    })
                    .unwrap_or(true);

                let check2 = min_bbox_cropping_ratio
                    .map(|min_bbox_cropping_ratio| {
                        let orig_area = bbox.area().raw();
                        let cropped_area = cropped.area();
                        cropped_area >= min_bbox_cropping_ratio * orig_area
                    })
                    .unwrap_or(true);

                (check1 && check2).then(|| RatioRectLabel {
                    rect: cropped.into(),
                    class: bbox.class,
                })
            })
            .collect_vec();

        Ok((cropped_image, cropped_bboxes))
    })
}
