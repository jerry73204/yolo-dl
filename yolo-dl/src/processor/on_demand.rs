//! The file caching implementation.

use crate::{common::*, profiling::Timing};

/// Image caching processor.
#[derive(Debug, Clone)]
pub struct OnDemandLoader {
    image_size: usize,
    image_channels: usize,
    device: Device,
}

impl OnDemandLoader {
    /// Build a new image caching processor.
    ///
    /// * `image_size` - The outcome image size in pixels.
    /// * `image_channels` - The expected number of image channels.
    /// * `device` - The outcome image device. It defaults to CPU if set to `None`.
    pub async fn new(
        image_size: usize,
        image_channels: usize,
        device: impl Into<Option<Device>>,
    ) -> Result<Self> {
        ensure!(image_size > 0, "image_size must be positive");
        ensure!(image_channels > 0, "image_channels must be positive");

        let loader = Self {
            image_size,
            image_channels,
            device: device.into().unwrap_or(Device::Cpu),
        };

        Ok(loader)
    }

    /// Load image and boxes.
    pub async fn load<P, B>(
        &self,
        image_path: P,
        orig_size: &PixelSize<usize>,
        bboxes: &[B],
    ) -> Result<(Tensor, Vec<RatioLabel>)>
    where
        P: AsRef<async_std::path::Path>,
        B: Borrow<PixelLabel>,
    {
        let Self {
            image_size,
            image_channels,
            device,
            ..
        } = *self;
        let image_path = image_path.as_ref();
        let [orig_h, orig_w] = orig_size.hw_params();

        ensure!(
            image_channels == 3,
            "image_channels other than 3 is not supported"
        );

        // compute cache size
        let resize_ratio =
            (image_size as f64 / orig_h as f64).min(image_size as f64 / orig_w as f64);
        let cache_h = (orig_h as f64 * resize_ratio) as usize;
        let cache_w = (orig_w as f64 * resize_ratio) as usize;
        let mut timing = Timing::new("cache_loader");

        // write cache if the cache is not valid
        let cached_image = {
            // load and resize image
            let image_path = image_path.to_owned();
            let (tensor, timing_) = async_std::task::spawn_blocking(move || -> Result<_> {
                tch::no_grad(|| -> Result<_> {
                    let image = vision::image::load(image_path)?;
                    {
                        let shape = image.size3()?;
                        let expect_shape = (image_channels as i64, orig_h as i64, orig_w as i64);
                        ensure!(
                            shape == expect_shape,
                            "image size does not match, expect {:?}, but get {:?}",
                            expect_shape,
                            shape
                        );
                    }
                    let image = image
                        // resize on cpu before moving to CUDA due to this issue
                        // https://github.com/LaurentMazare/tch-rs/issues/286
                        .resize2d_exact(cache_h as i64, cache_w as i64)?
                        .to_device(device)
                        .to_kind(Kind::Float)
                        .g_div1(255.0)
                        .set_requires_grad(false);
                    timing.add_event("load & resize");

                    Ok((image, timing))
                })
            })
            .await?;
            timing = timing_;

            tensor
        };

        // resize and center
        let top_pad = (image_size - cache_h) / 2;
        let bottom_pad = image_size - cache_h - top_pad;
        let left_pad = (image_size - cache_w) / 2;
        let right_pad = image_size - cache_w - left_pad;

        let output_image = tch::no_grad(|| {
            cached_image
                .view([1, image_channels as i64, cache_h as i64, cache_w as i64])
                .zero_pad2d(
                    left_pad as i64,
                    right_pad as i64,
                    top_pad as i64,
                    bottom_pad as i64,
                )
                .view([image_channels as i64, image_size as i64, image_size as i64])
                .set_requires_grad(false)
        });

        timing.add_event("pad");

        // compute new bboxes
        let output_bboxes: Vec<_> = {
            let image_size = R64::new(image_size as f64);
            bboxes
                .iter()
                .map(|orig_label| -> Result<_> {
                    let PixelLabel {
                        cycxhw: ref orig_bbox,
                        class,
                    } = *orig_label.borrow();

                    let resized_bbox = {
                        let resized_cy = orig_bbox.cy() * resize_ratio + top_pad as f64;
                        let resized_cx = orig_bbox.cx() * resize_ratio + left_pad as f64;
                        let resized_h = orig_bbox.h() * resize_ratio;
                        let resized_w = orig_bbox.w() * resize_ratio;
                        let resized_bbox = PixelCyCxHW::<R64>::from_cycxhw(
                            resized_cy, resized_cx, resized_h, resized_w,
                        )?;
                        resized_bbox
                    };

                    Ok(RatioLabel {
                        cycxhw: resized_bbox
                            .scale_to_unit(image_size.recip(), image_size.recip())?,
                        class,
                    })
                })
                .try_collect()?
        };

        timing.add_event("compute bboxes");

        timing.report();

        Ok((output_image, output_bboxes))
    }
}
