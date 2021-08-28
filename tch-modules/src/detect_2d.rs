use crate::common::*;

#[derive(Debug, Clone)]
pub struct Detect2DInit {
    pub num_classes: usize,
    pub anchors: Vec<RatioSize<R64>>,
}

impl Detect2DInit {
    pub fn build<'p, P>(self, path: P) -> Detect2D
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();
        let device = path.device();

        let Self {
            num_classes,
            anchors,
        } = self;

        Detect2D {
            num_classes,
            anchors,
            device,
            cache: None,
        }
    }
}

#[derive(Debug)]
pub struct Detect2D {
    num_classes: usize,
    anchors: Vec<RatioSize<R64>>,
    device: Device,
    cache: Option<Cache>,
}

impl Detect2D {
    pub fn forward(&mut self, tensor: &Tensor) -> Result<DenseDetectionTensor> {
        let Self {
            num_classes,
            ref anchors,
            ..
        } = *self;
        let (batch_size, channels, feature_h, feature_w) = tensor.size4()?;
        let anchors = anchors.to_owned();

        // load cached data
        let Cache {
            y_offsets,
            x_offsets,
            anchor_heights,
            anchor_widths,
            ..
        } = self.cache(tensor)?.shallow_clone();

        // compute outputs
        let num_anchors = anchors.len() as i64;
        let num_entries = num_classes as i64 + 5;
        debug_assert_eq!(channels, num_anchors * num_entries);

        // convert shape to [batch_size, n_entries, n_anchors, height, width]
        let outputs = tensor.view([batch_size, num_entries, num_anchors, feature_h, feature_w]);

        // positions in grid units
        let cy = (outputs.i((.., 0..1, .., .., ..)).sigmoid() * 2.0 - 0.5) / feature_h as f64
            + y_offsets.view([1, 1, 1, feature_h, 1]);
        let cx = (outputs.i((.., 1..2, .., .., ..)).sigmoid() * 2.0 - 0.5) / feature_w as f64
            + x_offsets.view([1, 1, 1, 1, feature_w]);

        debug_assert!({
            let expect_cy = {
                let array: Array5<f32> = outputs
                    .i((.., 0..1, .., .., ..))
                    .sigmoid()
                    .try_into_cv()
                    .unwrap();
                let mut expect_cy = array.clone();
                array
                    .indexed_iter()
                    .for_each(|((batch, entry, anchor, row, col), val)| {
                        expect_cy[(batch, entry, anchor, row, col)] =
                            (val * 2.0 - 0.5 + row as f32) / feature_h as f32;
                    });
                expect_cy
            };
            let expect_cx = {
                let array: Array5<f32> = outputs
                    .i((.., 1..2, .., .., ..))
                    .sigmoid()
                    .try_into_cv()
                    .unwrap();
                let mut expect_cx = array.clone();
                array
                    .indexed_iter()
                    .for_each(|((batch, entry, anchor, row, col), val)| {
                        expect_cx[(batch, entry, anchor, row, col)] =
                            (val * 2.0 - 0.5 + col as f32) / feature_w as f32;
                    });
                expect_cx
            };

            let cy: Array5<f32> = (&cy).try_into_cv().unwrap();
            let cx: Array5<f32> = (&cx).try_into_cv().unwrap();

            cy.indexed_iter()
                .all(|(index, &actual)| abs_diff_eq!(actual, expect_cy[index]))
                && cx
                    .indexed_iter()
                    .all(|(index, &actual)| abs_diff_eq!(actual, expect_cx[index]))
        });

        // bbox sizes in grid units
        let h = outputs
            .i((.., 2..3, .., .., ..))
            .sigmoid()
            .mul(2.0)
            .pow(2.0)
            * anchor_heights.view([1, 1, num_anchors, 1, 1]);
        let w = outputs
            .i((.., 3..4, .., .., ..))
            .sigmoid()
            .mul(2.0)
            .pow(2.0)
            * anchor_widths.view([1, 1, num_anchors, 1, 1]);

        // objectness
        let obj_logit = outputs.i((.., 4..5, .., .., ..));

        // sparse classification
        let class_logit = outputs.i((.., 5.., .., .., ..));

        Ok(DenseDetectionTensorUnchecked {
            cy,
            cx,
            h,
            w,
            obj_logit,
            class_logit,
            anchors,
        }
        .try_into()
        .unwrap())
    }

    pub fn anchors(&self) -> &[RatioSize<R64>] {
        &self.anchors
    }

    fn cache(&mut self, tensor: &Tensor) -> Result<&Cache> {
        tch::no_grad(move || -> Result<_> {
            let Self {
                device,
                ref anchors,
                ref mut cache,
                ..
            } = *self;

            let (_b, _c, feature_h, feature_w) = tensor.size4()?;
            let expect_size = GridSize::from_hw(feature_h, feature_w).unwrap();

            let is_hit = cache
                .as_ref()
                .map(|cache| cache.feature_size == expect_size)
                .unwrap_or(false);

            if !is_hit {
                // print anchor sizes in grid unit
                for (index, ratio_size) in anchors.iter().enumerate() {
                    let grid_h = ratio_size.h * feature_h as f64;
                    let grid_w = ratio_size.w * feature_w as f64;
                    info!("anchro sizes for Detect2D");
                    info!("  - feature size\t{}x{}", feature_h, feature_w);
                    info!(
                        "  - anchor size {}\t{}x{} (ratio) => {}x{} (grid)",
                        index, ratio_size.h, ratio_size.w, grid_h, grid_w
                    );
                }

                let y_offsets = (Tensor::arange(feature_h, (Kind::Float, device))
                    / feature_h as f64)
                    .set_requires_grad(false);
                let x_offsets = (Tensor::arange(feature_w, (Kind::Float, device))
                    / feature_w as f64)
                    .set_requires_grad(false);

                let (anchor_heights, anchor_widths) = {
                    let (anchor_h_vec, anchor_w_vec) = anchors
                        .iter()
                        .cloned()
                        .map(|anchor_size| {
                            let anchor_size = anchor_size.cast::<f32>().unwrap();
                            (anchor_size.h, anchor_size.w)
                        })
                        .unzip_n_vec();

                    let anchor_heights = Tensor::of_slice(&anchor_h_vec)
                        .set_requires_grad(false)
                        .to_device(device);
                    let anchor_widths = Tensor::of_slice(&anchor_w_vec)
                        .set_requires_grad(false)
                        .to_device(device);

                    (anchor_heights, anchor_widths)
                };

                let new_cache = Cache {
                    feature_size: expect_size,
                    y_offsets,
                    x_offsets,
                    anchor_heights,
                    anchor_widths,
                };

                *cache = Some(new_cache);
            }

            let cache = cache.as_ref().unwrap();
            Ok(cache)
        })
    }
}

#[derive(Debug)]
struct Cache {
    feature_size: GridSize<i64>,
    y_offsets: Tensor,
    x_offsets: Tensor,
    anchor_heights: Tensor,
    anchor_widths: Tensor,
}
