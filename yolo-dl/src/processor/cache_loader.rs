//! The data caching implementation.

use crate::{common::*, profiling::Timing};

// cache loader

/// Image caching processor.
#[derive(Debug, Clone)]
pub struct CacheLoader {
    cache_dir: async_std::path::PathBuf,
    image_size: usize,
    image_channels: usize,
    device: Device,
}

impl CacheLoader {
    /// Build a new image caching processor.
    ///
    /// * `cache_dir` - The directory to store caches.
    /// * `image_size` - The outcome image size in pixels.
    /// * `image_channels` - The expected number of image channels.
    /// * `device` - The outcome image device. It defaults to CPU if set to `None`.
    pub async fn new<P>(
        cache_dir: P,
        image_size: usize,
        image_channels: usize,
        device: impl Into<Option<Device>>,
    ) -> Result<Self>
    where
        P: AsRef<async_std::path::Path>,
    {
        ensure!(image_size > 0, "image_size must be positive");
        ensure!(image_channels > 0, "image_channels must be positive");

        let cache_dir = cache_dir.as_ref().to_owned();
        async_std::fs::create_dir_all(&*cache_dir).await?;

        let loader = Self {
            cache_dir,
            image_size,
            image_channels,
            device: device.into().unwrap_or(Device::Cpu),
        };

        Ok(loader)
    }

    /// Load image and boxes.
    ///
    /// If cache hit, it loads the cached image and boxes from cache directory.
    /// If cache miss, it loads and resizes the image from original path and saves it to cache directory.
    pub async fn load_cache<P, B>(
        &self,
        image_path: P,
        orig_size: &PixelSize<usize>,
        bboxes: &[B],
    ) -> Result<(Tensor, Vec<LabeledRatioCyCxHW>)>
    where
        P: AsRef<async_std::path::Path>,
        B: Borrow<LabeledPixelCyCxHW<R64>>,
    {
        use async_std::{fs::File, io::BufWriter};

        let Self {
            image_size,
            image_channels,
            device,
            ..
        } = *self;
        let image_path = image_path.as_ref();
        let PixelSize {
            h: orig_h,
            w: orig_w,
            ..
        } = *orig_size;

        ensure!(
            image_channels == 3,
            "image_channels other than 3 is not supported"
        );

        // compute cache size
        let resize_ratio =
            (image_size as f64 / orig_h as f64).min(image_size as f64 / orig_w as f64);
        let cache_h = (orig_h as f64 * resize_ratio) as usize;
        let cache_w = (orig_w as f64 * resize_ratio) as usize;
        let cache_components = cache_h * cache_w * image_channels;
        let cache_bytes = cache_components * mem::size_of::<f32>();
        let mut timing = Timing::new("cache_loader");

        // construct cache path
        let cache_path = self.cache_dir.join(format!(
            "{}-{}-{}-{}",
            percent_encoding::utf8_percent_encode(image_path.to_str().unwrap(), NON_ALPHANUMERIC),
            image_channels,
            cache_h,
            cache_w,
        ));

        // check if the cache is valid
        let is_valid = if cache_path.is_file().await {
            let image_modified = image_path.metadata().await?.modified()?;
            let cache_meta = cache_path.metadata().await?;
            let cache_modified = cache_meta.modified()?;
            cache_modified > image_modified && cache_meta.len() == cache_bytes as u64
        } else {
            false
        };

        timing.add_event("check cache validity");

        // write cache if the cache is not valid
        let cached_image = if is_valid {
            // load from cache
            let image = async_std::task::spawn_blocking(move || -> Result<_> {
                let image = Tensor::f_from_file(
                    cache_path.to_str().unwrap(),
                    false,
                    Some(cache_components as i64),
                    (Kind::Float, Device::Cpu),
                )?
                .to_device(device)
                .view_(&[image_channels as i64, cache_h as i64, cache_w as i64]);

                Ok(image)
            })
            .await?;

            timing.add_event("load cache");

            image
        } else {
            // load and resize image
            let image_path = image_path.to_owned();
            let (tensor, buffer, timing_) =
                async_std::task::spawn_blocking(move || -> Result<_> {
                    tch::no_grad(|| -> Result<_> {
                        let image = vision::image::load(image_path)?;
                        {
                            let shape = image.size3()?;
                            let expect_shape =
                                (image_channels as i64, orig_h as i64, orig_w as i64);
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

                        let mut buffer = vec![0; cache_bytes];
                        image.copy_data_u8(&mut buffer, cache_components);
                        timing.add_event("rehape & copy");

                        Ok((image, buffer, timing))
                    })
                })
                .await?;
            timing = timing_;

            // write cache
            let mut writer = BufWriter::new(File::create(&cache_path).await?);
            writer.write_all(&buffer).await?;
            writer.flush().await?; // make sure the file is ready in the next cache hit

            timing.add_event("write cache");

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
                    let LabeledPixelCyCxHW {
                        bbox: ref orig_bbox,
                        category_id,
                    } = *orig_label.borrow();

                    let resized_bbox = {
                        // let [orig_cy, orig_cx, orig_h, orig_w] = orig_bbox.cycxhw();
                        let resized_cy = orig_bbox.cy() * resize_ratio + top_pad as f64;
                        let resized_cx = orig_bbox.cx() * resize_ratio + left_pad as f64;
                        let resized_h = orig_bbox.h() * resize_ratio;
                        let resized_w = orig_bbox.w() * resize_ratio;
                        let resized_bbox = PixelCyCxHW::<R64>::from_cycxhw(
                            resized_cy, resized_cx, resized_h, resized_w,
                        )?;
                        resized_bbox
                    };

                    Ok(LabeledRatioCyCxHW {
                        bbox: resized_bbox.to_ratio_unit(image_size, image_size)?,
                        category_id,
                    })
                })
                .try_collect()?
        };

        timing.add_event("compute bboxes");

        timing.report();

        Ok((output_image, output_bboxes))
    }
}
