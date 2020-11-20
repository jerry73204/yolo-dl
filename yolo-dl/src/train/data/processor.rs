use super::DataRecord;
use crate::{common::*, util::Timing};

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

// cache loader

#[derive(Debug, Clone)]
pub struct CacheLoader {
    cache_dir: async_std::path::PathBuf,
    image_size: usize,
    image_channels: usize,
}

impl CacheLoader {
    pub async fn new<P>(cache_dir: P, image_size: usize, image_channels: usize) -> Result<Self>
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
        };

        Ok(loader)
    }

    pub async fn load_cache(&self, record: &DataRecord) -> Result<(Tensor, Vec<LabeledRatioBBox>)> {
        use async_std::{fs::File, io::BufWriter};

        // load config
        let Self {
            image_size,
            image_channels,
            ..
        } = *self;
        let DataRecord {
            path: ref image_path,
            size:
                PixelSize {
                    height: orig_height,
                    width: orig_width,
                    ..
                },
            ref bboxes,
        } = *record;
        let image_path: &async_std::path::Path = (**image_path).as_ref();

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
        let mut timing = Timing::new();

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
                    FLOAT_CPU,
                )?
                .view_(&[
                    image_channels as i64,
                    cache_height as i64,
                    cache_width as i64,
                ])
                .requires_grad_(false);

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
                        .to_rgb()
                        .into_flat_samples();
                    debug_assert_eq!(samples.len(), cache_height * cache_width * image_channels);

                    timing.set_record("load raw & resize");

                    let tensor = tch::no_grad(|| {
                        Tensor::of_slice(&samples)
                            .to_kind(Kind::Float)
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

        let output_image = {
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
        };

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
                } = *orig_bbox;

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

        // info!("{:#?}", timing.records());

        Ok((output_image, output_bboxes))
    }
}
