//! The file caching implementation.

use crate::{common::*, label::Label};
use bbox::{RectNum as _, Transform, HW, TLBR};
use percent_encoding::NON_ALPHANUMERIC;
use tch_goodies::{Pixel, Ratio, TensorExt as _};

/// Image caching processor.
#[derive(Debug, Clone)]
pub struct FileCache {
    cache_dir: async_std::path::PathBuf,
    image_size: usize,
    image_channels: usize,
    device: Device,
}

impl FileCache {
    /// Build a new image caching processor.
    ///
    /// * `cache_dir` - The directory to store caches.
    /// * `image_size` - The outcome image size in pixels.
    /// * `image_channels` - The expected number of image channels.
    /// * `device` - The outcome image device. It defaults to CPU if set to `None`.
    pub async fn new(
        cache_dir: impl AsRef<async_std::path::Path>,
        image_size: usize,
        image_channels: usize,
        device: impl Into<Option<Device>>,
    ) -> Result<Self> {
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
    pub async fn load_cache(
        &self,
        image_path: impl AsRef<async_std::path::Path>,
        orig_size: &Pixel<HW<usize>>,
        bboxes: impl IntoIterator<Item = impl Borrow<Pixel<Label>>>,
    ) -> Result<(Tensor, Vec<Ratio<Label>>)> {
        use async_std::{fs::File, io::BufWriter};

        let Self {
            image_size,
            image_channels,
            device,
            ..
        } = *self;
        let image_path = image_path.as_ref();
        let [orig_h, orig_w] = orig_size.hw();

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

        // write cache if the cache is not valid
        let cached_image = if is_valid {
            // load from cache
            let image = async_std::task::spawn_blocking(move || -> Result<_> {
                // BUG: when the dataset is small, the file opening can race with
                // cache file saving.
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

            image
        } else {
            // load and resize image
            let image_path = image_path.to_owned();
            let (tensor, buffer) = async_std::task::spawn_blocking(move || -> Result<_> {
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
                        .g_div_scalar(255.0)
                        .set_requires_grad(false);

                    let mut buffer = vec![0; cache_bytes];
                    image.copy_data_u8(&mut buffer, cache_components);

                    Ok((image, buffer))
                })
            })
            .await?;

            // write cache
            let mut writer = BufWriter::new(File::create(&cache_path).await?);
            writer.write_all(&buffer).await?;
            writer.flush().await?; // make sure the file is ready in the next cache hit

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

        // compute new bboxes
        let transform = {
            let src = TLBR::from_tlbr([0.0, 0.0, orig_h as f64, orig_w as f64]);
            let tgt = TLBR::from_tlhw([
                top_pad as f64,
                left_pad as f64,
                cache_h as f64,
                cache_w as f64,
            ]);
            Pixel(Transform::from_rects(&src, &tgt).cast::<R64>())
        };

        let output_bboxes: Vec<_> = {
            let image_size = r64(image_size as f64);
            bboxes
                .into_iter()
                .map(|orig_label| -> Result<_> {
                    let orig_label = orig_label.borrow();
                    let new_label = Ratio(Label {
                        rect: (&transform.0 * &orig_label.rect).scale(image_size.recip()),
                        class: orig_label.class,
                    });
                    Ok(new_label)
                })
                .try_collect()?
        };

        Ok((output_image, output_bboxes))
    }
}
