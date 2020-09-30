use crate::{common::*, config::Config, message::LoggingMessage};

// utility funcions

pub fn crop_image(image: &Tensor, top: Ratio, bottom: Ratio, left: Ratio, right: Ratio) -> Tensor {
    assert!(left < right);
    assert!(top < bottom);

    let (_channels, height, width) = image.size3().unwrap();
    let height = height as f64;
    let width = width as f64;

    let top = (f64::from(top) * height) as i64;
    let bottom = (f64::from(bottom) * height) as i64;
    let left = (f64::from(left) * width) as i64;
    let right = (f64::from(right) * width) as i64;

    let cropped = image.i((.., top..bottom, left..right));
    cropped
}

pub fn resize_image(input: &Tensor, out_w: i64, out_h: i64) -> Fallible<Tensor> {
    let (_channels, _height, _width) = input.size3().unwrap();

    let resized = vision::image::resize_preserve_aspect_ratio(
        &(input * 255.0).to_kind(Kind::Uint8),
        out_w,
        out_h,
    )?
    .to_kind(Kind::Float)
        / 255.0;

    Ok(resized)
}

pub fn batch_resize_image(input: &Tensor, out_w: i64, out_h: i64) -> Fallible<Tensor> {
    let (bsize, _channels, _height, _width) = input.size4().unwrap();

    let input_scaled = (input * 255.0).to_kind(Kind::Uint8);
    let resized_vec = (0..bsize)
        .map(|index| {
            let resized = vision::image::resize_preserve_aspect_ratio(
                &input_scaled.select(0, index),
                out_w,
                out_h,
            )?;
            Ok(resized)
        })
        .collect::<Fallible<Vec<_>>>()?;
    let resized = Tensor::stack(resized_vec.as_slice(), 0);
    let resized_scaled = resized.to_kind(Kind::Float) / 255.0;

    Ok(resized_scaled)
}

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

    pub async fn load_cache<P>(
        &self,
        image_path: P,
        orig_height: usize,
        orig_width: usize,
    ) -> Result<Tensor>
    where
        P: AsRef<async_std::path::Path>,
    {
        use async_std::{fs::File, io::BufWriter};

        // load config
        let Self {
            image_size,
            image_channels,
            ..
        } = *self;

        let image_path = image_path.as_ref().to_owned();
        ensure!(image_channels == 3, "image_channels != 3 is not supported");

        // compute cache size
        let scale =
            (image_size as f64 / orig_height as f64).min(image_size as f64 / orig_width as f64);
        let resize_height = (orig_height as f64 * scale) as usize;
        let resize_width = (orig_width as f64 * scale) as usize;
        let component_size = mem::size_of::<f32>();
        let num_components = resize_height * resize_width * image_channels;
        let cache_size = num_components * component_size;

        // construct cache path
        let cache_path = self.cache_dir.join(format!(
            "{}-{}-{}-{}",
            percent_encoding::utf8_percent_encode(image_path.to_str().unwrap(), NON_ALPHANUMERIC),
            image_channels,
            resize_height,
            resize_width,
        ));

        // check if the cache is valid
        let is_valid = if cache_path.is_file().await {
            let image_modified = image_path.metadata().await?.modified()?;
            let cache_meta = cache_path.metadata().await?;
            let cache_modified = cache_meta.modified()?;

            if cache_modified > image_modified {
                cache_meta.len() == cache_size as u64
            } else {
                false
            }
        } else {
            false
        };

        // write cache if the cache is not valid
        if !is_valid {
            // load and resize image
            let array = async_std::task::spawn_blocking(move || -> Result<_> {
                let image = image::io::Reader::open(&image_path)
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
                    })?;
                let image = image
                    .resize_exact(
                        resize_width as u32,
                        resize_height as u32,
                        FilterType::CatmullRom,
                    )
                    .to_rgb();

                // convert to channel first
                let array = Array3::from_shape_fn(
                    [image_channels, resize_height, resize_width],
                    |(channel, row, col)| {
                        let component = image.get_pixel(col as u32, row as u32).channels()[channel];
                        component as f32 / 255.0
                    },
                );

                Ok(array)
            })
            .await?;

            // write cache
            let mut writer = BufWriter::new(File::create(&cache_path).await?);
            for bytes in array.map(|component| component.to_le_bytes()).iter() {
                writer.write_all(bytes).await?;
            }
            writer.flush().await?;
        };

        // load from cache
        let image = async_std::task::spawn_blocking(move || -> Result<_> {
            let image = Tensor::f_from_file(
                cache_path.to_str().unwrap(),
                false,
                Some(num_components as i64),
                FLOAT_CPU,
            )?
            .view_(&[
                image_channels as i64,
                resize_height as i64,
                resize_width as i64,
            ])
            .requires_grad_(false);

            Ok(image)
        })
        .await?;

        Ok(image)
    }
}

// mosaic processor

pub struct MosaicProcessor {
    image_size: i64,
    mosaic_margin: Ratio,
}

impl MosaicProcessor {
    pub fn new(image_size: i64, mosaic_margin: Ratio) -> Self {
        Self {
            image_size,
            mosaic_margin,
        }
    }

    pub async fn make_mosaic(
        &self,
        bbox_image_vec: Vec<(Vec<LabeledRatioBBox>, Tensor)>,
    ) -> Fallible<(Vec<LabeledRatioBBox>, Tensor)> {
        let Self {
            image_size,
            mosaic_margin,
        } = *self;
        let mosaic_margin = mosaic_margin.raw();

        debug_assert_eq!(bbox_image_vec.len(), 4);
        let mut rng = StdRng::from_entropy();

        // random select pivot point
        let pivot_row: Ratio = rng.gen_range(mosaic_margin, 1.0 - mosaic_margin).into();
        let pivot_col: Ratio = rng.gen_range(mosaic_margin, 1.0 - mosaic_margin).into();

        // crop images
        let ranges = vec![
            (0.0.into(), pivot_row, 0.0.into(), pivot_col),
            (0.0.into(), pivot_row, pivot_col, 1.0.into()),
            (pivot_row, 1.0.into(), 0.0.into(), pivot_col),
            (pivot_row, 1.0.into(), pivot_col, 1.0.into()),
        ];

        let mut crop_iter = futures::stream::iter(
            bbox_image_vec.into_iter().zip_eq(ranges.into_iter()),
        )
        .map(move |((bboxes, image), (top, bottom, left, right))| {
            // sanity check
            {
                let (_c, h, w) = image.size3().unwrap();
                debug_assert_eq!(h, image_size);
                debug_assert_eq!(w, image_size);
            }

            let cropped_image = crop_image(&image, top, bottom, left, right);
            let cropped_bboxes: Vec<_> = bboxes
                .into_iter()
                .filter_map(|bbox| bbox.crop(top, bottom, left, right))
                .collect();

            (cropped_bboxes, cropped_image)
        });

        let (bboxes_tl, cropped_tl) = crop_iter.next().await.unwrap();
        let (bboxes_tr, cropped_tr) = crop_iter.next().await.unwrap();
        let (bboxes_bl, cropped_bl) = crop_iter.next().await.unwrap();
        let (bboxes_br, cropped_br) = crop_iter.next().await.unwrap();
        debug_assert_eq!(crop_iter.next().await, None);

        // merge cropped images
        let (merged_bboxes, merged_image) = async_std::task::spawn_blocking(move || {
            let merged_top = Tensor::cat(&[cropped_tl, cropped_tr], 2);

            let merged_bottom = Tensor::cat(&[cropped_bl, cropped_br], 2);

            let merged_image = Tensor::cat(&[merged_top, merged_bottom], 1);
            debug_assert_eq!(merged_image.size3().unwrap(), (3, image_size, image_size));

            // merge cropped bboxes
            let merged_bboxes: Vec<_> = bboxes_tl
                .into_iter()
                .chain(bboxes_tr.into_iter())
                .chain(bboxes_bl.into_iter())
                .chain(bboxes_br.into_iter())
                .collect();

            (merged_bboxes, merged_image)
        })
        .await;

        Ok((merged_bboxes, merged_image))
    }
}
