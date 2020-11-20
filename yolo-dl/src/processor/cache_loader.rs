use crate::{common::*, profiling::Timing};

// cache loader

#[derive(Debug, Clone)]
pub struct CacheLoader {
    cache_dir: async_std::path::PathBuf,
    image_size: usize,
    image_channels: usize,
    device: Device,
}

impl CacheLoader {
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

    pub async fn load_cache<P, B>(
        &self,
        image_path: P,
        bboxes: &[B],
    ) -> Result<(Tensor, Vec<LabeledRatioBBox>)>
    where
        P: AsRef<async_std::path::Path>,
        B: Borrow<LabeledPixelBBox<R64>>,
    {
        use async_std::{fs::File, io::BufWriter};

        let Self {
            image_size,
            image_channels,
            device,
            ..
        } = *self;
        let image_path = image_path.as_ref();

        let (orig_height, orig_width) = {
            let meta = immeta::load_from_file(image_path)?;
            let immeta::Dimensions { height, width } = meta.dimensions();
            (height, width)
        };

        ensure!(
            image_channels == 3,
            "image_channels other than 3 is not supported"
        );

        // compute cache size
        let resize_ratio =
            (image_size as f64 / orig_height as f64).min(image_size as f64 / orig_width as f64);
        let cache_height = (orig_height as f64 * resize_ratio) as usize;
        let cache_width = (orig_width as f64 * resize_ratio) as usize;
        let cache_components = cache_height * cache_width * image_channels;
        let cache_bytes = cache_components * mem::size_of::<f32>();
        let mut timing = Timing::new("cache loader");

        // construct cache path
        let cache_path = self.cache_dir.join(format!(
            "{}-{}-{}-{}",
            percent_encoding::utf8_percent_encode(image_path.to_str().unwrap(), NON_ALPHANUMERIC),
            image_channels,
            cache_height,
            cache_width,
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

        timing.set_record("check cache validity");

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
                .view_(&[
                    image_channels as i64,
                    cache_height as i64,
                    cache_width as i64,
                ]);

                Ok(image)
            })
            .await?;

            timing.set_record("load cache");

            image
        } else {
            // load and resize image
            let image_path = image_path.to_owned();
            let (tensor, buffer, timing_) =
                async_std::task::spawn_blocking(move || -> Result<_> {
                    let FlatSamples { samples, .. } = image::io::Reader::open(&image_path)
                        .with_context(|| format!("failed to open {}", image_path.display()))?
                        .with_guessed_format()
                        .with_context(|| {
                            format!(
                                "failed to determine the image file format: {}",
                                image_path.display()
                            )
                        })?
                        .decode()
                        .with_context(|| {
                            format!("failed to decode image file: {}", image_path.display())
                        })?
                        .resize_exact(
                            cache_width as u32,
                            cache_height as u32,
                            FilterType::CatmullRom,
                        )
                        .to_rgb8()
                        .into_flat_samples();
                    debug_assert_eq!(samples.len(), cache_height * cache_width * image_channels);

                    timing.set_record("load raw & resize");

                    let tensor = tch::no_grad(|| {
                        Tensor::of_slice(&samples)
                            .to_kind(Kind::Float)
                            .to_device(device)
                            .g_div1(255.0)
                            .view([
                                cache_height as i64,
                                cache_width as i64,
                                image_channels as i64,
                            ])
                            .permute(&[2, 0, 1])
                            .set_requires_grad(false)
                    });

                    let mut buffer = vec![0; cache_bytes];
                    tensor.copy_data_u8(&mut buffer, cache_components);
                    timing.set_record("rehape & copy");

                    Ok((tensor, buffer, timing))
                })
                .await?;
            timing = timing_;

            // write cache
            let mut writer = BufWriter::new(File::create(&cache_path).await?);
            writer.write_all(&buffer).await?;
            writer.flush().await?; // make sure the file is ready in the next cache hit

            timing.set_record("write cache");

            tensor
        };

        // resize and center
        let top_pad = (image_size - cache_height) / 2;
        let bottom_pad = image_size - cache_height - top_pad;
        let left_pad = (image_size - cache_width) / 2;
        let right_pad = image_size - cache_width - left_pad;

        let output_image = tch::no_grad(|| {
            cached_image
                .view([
                    1,
                    image_channels as i64,
                    cache_height as i64,
                    cache_width as i64,
                ])
                .zero_pad2d(
                    left_pad as i64,
                    right_pad as i64,
                    top_pad as i64,
                    bottom_pad as i64,
                )
                .view([image_channels as i64, image_size as i64, image_size as i64])
                .set_requires_grad(false)
        });

        timing.set_record("pad");

        // compute new bboxes
        let output_bboxes: Vec<_> = bboxes
            .iter()
            .map(|orig_bbox| -> Result<_> {
                let LabeledPixelBBox {
                    bbox:
                        PixelBBox {
                            cycxhw: [orig_cy, orig_cx, orig_h, orig_w],
                            ..
                        },
                    category_id,
                } = *orig_bbox.borrow();

                let new_cy = orig_cy * resize_ratio + top_pad as f64;
                let new_cx = orig_cx * resize_ratio + left_pad as f64;
                let new_h = orig_h * resize_ratio;
                let new_w = orig_w * resize_ratio;

                let bbox = match PixelBBox::from_cycxhw([new_cy, new_cx, new_h, new_w])
                    .to_ratio_bbox(image_size, image_size)
                {
                    Ok(bbox) => bbox,
                    Err(bbox) => {
                        warn!("invalid bbox found in '{}'", image_path.display());
                        bbox
                    }
                };

                Ok(LabeledRatioBBox { bbox, category_id })
            })
            .try_collect()?;

        timing.set_record("compute bboxes");

        timing.report();

        Ok((output_image, output_bboxes))
    }
}
