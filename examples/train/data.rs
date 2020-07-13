use crate::{common::*, config::Config, message::LoggingMessage};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Record {
    pub path: PathBuf,
    pub height: usize,
    pub width: usize,
    pub bboxes: Vec<BBox>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BBox {
    pub xywh: [R64; 4],
    pub category_id: usize,
}

#[derive(Debug, TensorLike)]
pub struct TrainingRecord {
    pub epoch: usize,
    pub step: usize,
    pub image: Tensor,
    #[tensor_like(clone)]
    pub bboxes: Vec<BBox>,
}

pub struct DataSet {
    config: Arc<Config>,
    dataset: coco::DataSet,
}

impl DataSet {
    pub async fn new(config: Arc<Config>) -> Result<Self> {
        let Config {
            dataset_dir,
            dataset_name,
            ..
        } = &*config;
        let dataset = coco::DataSet::load_async(dataset_dir, &dataset_name).await?;

        Ok(Self { config, dataset })
    }

    pub fn input_channels(&self) -> usize {
        3
    }

    pub fn num_classes(&self) -> usize {
        self.dataset.instances.categories.len()
    }

    pub async fn train_stream(
        &self,
        logging_tx: broadcast::Sender<LoggingMessage>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<TrainingRecord>> + Send>>> {
        let Config {
            ref dataset_dir,
            ref dataset_name,
            ref cache_dir,
            image_size,
            mosaic_margin,
            mini_batch_size,
            ..
        } = *self.config;
        let mosaic_margin = mosaic_margin.raw();
        let image_size = image_size.get() as i64;

        let coco::DataSet {
            instances,
            image_dir,
            ..
        } = &self.dataset;

        let annotations = instances
            .annotations
            .iter()
            .map(|ann| (ann.id, ann))
            .collect::<HashMap<_, _>>();
        let images = instances
            .images
            .iter()
            .map(|img| (img.id, img))
            .collect::<HashMap<_, _>>();
        let categories = instances
            .categories
            .iter()
            .map(|cat| (cat.id, cat))
            .collect::<HashMap<_, _>>();

        let records = Arc::new(
            annotations
                .iter()
                .map(|(id, ann)| (ann.image_id, ann))
                .into_group_map()
                .into_iter()
                .map(|(image_id, anns)| {
                    let image = &images[&image_id];
                    let bboxes = anns
                        .into_iter()
                        .map(|ann| BBox {
                            xywh: ann.bbox.clone(),
                            category_id: ann.category_id,
                        })
                        .collect::<Vec<_>>();

                    Record {
                        path: image_dir.join(&image.file_name),
                        height: image.height,
                        width: image.width,
                        bboxes,
                    }
                })
                .map(Arc::new)
                .collect::<Vec<_>>(),
        );

        // repeat records
        let stream = futures::stream::repeat(records).enumerate();

        // sample 4 records per step
        let stream = stream.flat_map(|(epoch, records)| {
            let mut rng = rand::thread_rng();
            let num_records = records.len();

            let mut index_iters: Vec<_> = (0..4)
                .map(|_| {
                    let mut indexes: Vec<_> = (0..num_records).collect();
                    indexes.shuffle(&mut rng);
                    indexes.into_iter()
                })
                .collect();

            let record_vec_iter = (0..num_records)
                .map(|_| {
                    let record_vec: Vec<Arc<Record>> = index_iters
                        .iter_mut()
                        .map(|iter| iter.next().unwrap())
                        .map(|index| records[index].clone())
                        .collect();
                    (epoch, record_vec)
                })
                .collect::<Vec<_>>();

            futures::stream::iter(record_vec_iter)
        });

        // add step count
        let stream = stream
            .enumerate()
            .map(|(step, (epoch, record_vec))| Ok((step, epoch, record_vec)));

        // start of unordered maps
        let stream = stream.try_overflowing_enumerate();

        // load and cache images
        let cache_loader = Arc::new(CacheLoader::new(&cache_dir, image_size as usize, 3).await?);

        let stream = stream.try_par_then_unordered(None, move |(index, args)| {
            let (step, epoch, record_vec) = args;
            let cache_loader = cache_loader.clone();

            async move {
                let load_cache_futs = record_vec
                    .into_iter()
                    .map(|record| {
                        let cache_loader = cache_loader.clone();

                        async move {
                            let Record {
                                ref path,
                                height,
                                width,
                                ..
                            } = *record;

                            // load cache
                            let image = cache_loader.load_cache(path, height, width).await?;

                            // extend to specified size
                            let resized = resize_image(&image, image_size, image_size)?;

                            Fallible::Ok((record, resized))
                        }
                    })
                    .map(async_std::task::spawn);

                let record_image_vec: Vec<(_, _)> =
                    futures::future::try_join_all(load_cache_futs).await?;

                Fallible::Ok((index, (step, epoch, record_image_vec)))
            }
        });

        // make mosaic
        let stream = stream.try_par_then_unordered(None, move |(index, args)| async move {
            let (step, epoch, record_image_vec) = args;
            let (merged_bboxes, merged_image) =
                make_mosaic(record_image_vec, image_size, mosaic_margin).await?;

            Fallible::Ok((index, (step, epoch, merged_bboxes, merged_image)))
        });

        // map to output type
        let stream = stream.try_par_then_unordered(None, |(index, args)| async move {
            let (step, epoch, bboxes, image) = args;
            let record = TrainingRecord {
                epoch,
                step,
                image,
                bboxes,
            };

            Ok((index, record))
        });

        // reorder items
        let stream = stream.try_reorder_enumerated();

        let stream = Box::pin(stream);
        Ok(stream)
    }
}

#[derive(Debug, Clone)]
pub struct AugmentationProcessor {
    // TODO
}

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
            let image = async_std::task::spawn_blocking(move || -> Result<_> {
                let image = image::open(&image_path)
                    .with_context(|| format!("failed to open {}", image_path.display()))?
                    .resize_exact(
                        resize_width as u32,
                        resize_height as u32,
                        FilterType::CatmullRom,
                    )
                    .to_rgb();
                Ok(image)
            })
            .await?;

            // convert to channel first
            let array = Array3::from_shape_fn(
                [image_channels, resize_height, resize_width],
                |(channel, row, col)| {
                    let component = image.get_pixel(col as u32, row as u32).channels()[channel];
                    component as f32 / 255.0
                },
            );

            // write cache
            let mut writer = BufWriter::new(File::create(&cache_path).await?);
            for bytes in array.map(|component| component.to_le_bytes()).iter() {
                writer.write_all(bytes).await?;
            }
            writer.flush().await?;
        };

        // load from cache
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
    }
}

pub fn crop_image(image: &Tensor, top: i64, bottom: i64, left: i64, right: i64) -> Tensor {
    debug_assert!(top >= 0);
    debug_assert!(bottom >= 0);
    debug_assert!(left >= 0);
    debug_assert!(right >= 0);
    debug_assert!(left < right);
    debug_assert!(top < bottom);
    let (channels, height, width) = image.size3().unwrap();
    let cropped = image.i((.., top..bottom, left..right));
    cropped
}

pub fn crop_bbox(bbox: &BBox, top: i64, bottom: i64, left: i64, right: i64) -> Option<BBox> {
    let BBox {
        xywh: [bbox_left, bbox_top, w, h],
        category_id,
    } = *bbox;
    let bbox_right = bbox_left + w;
    let bbox_bottom = bbox_top + h;

    let crop_top = bbox_left.raw().max(top as f64);
    let crop_bottom = bbox_right.raw().max(bottom as f64);
    let crop_left = bbox_left.raw().max(left as f64);
    let crop_right = bbox_right.raw().max(right as f64);

    let crop_width = crop_right - crop_left;
    let crop_height = crop_bottom - crop_top;

    if crop_height >= 1.0 && crop_width >= 1.0 {
        Some(BBox {
            xywh: [
                R64::new(crop_left),
                R64::new(crop_top),
                R64::new(crop_width),
                R64::new(crop_height),
            ],
            category_id,
        })
    } else {
        None
    }
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

pub async fn make_mosaic(
    record_image_vec: Vec<(Arc<Record>, Tensor)>,
    image_size: i64,
    mosaic_margin: f64,
) -> Fallible<(Vec<BBox>, Tensor)> {
    debug_assert_eq!(record_image_vec.len(), 4);
    let mut rng = StdRng::from_entropy();

    // random select pivot point
    let pivot_row = (rng.gen_range(mosaic_margin, 1.0 - mosaic_margin) * image_size as f64) as i64;
    let pivot_col = (rng.gen_range(mosaic_margin, 1.0 - mosaic_margin) * image_size as f64) as i64;

    // crop images
    let ranges = vec![
        (0, pivot_row, 0, pivot_col),
        (0, pivot_row, pivot_col, image_size),
        (pivot_row, image_size, 0, pivot_col),
        (pivot_row, image_size, pivot_col, image_size),
    ];

    let mut record_cropped_iter =
        futures::stream::iter(record_image_vec.into_iter().zip_eq(ranges.into_iter())).par_then(
            None,
            move |((record, image), (top, bottom, left, right))| async move {
                let Record {
                    ref path,
                    height,
                    width,
                    ref bboxes,
                } = *record;

                {
                    let (_c, h, w) = image.size3().unwrap();
                    debug_assert_eq!(h, image_size);
                    debug_assert_eq!(w, image_size);
                }

                let cropped_image = crop_image(&image, top, bottom, left, right);
                let cropped_bboxes: Vec<_> = bboxes
                    .into_iter()
                    .filter_map(|bbox| crop_bbox(bbox, top, bottom, left, right))
                    .collect();

                let cropped_record = Record {
                    path: path.clone(),
                    height,
                    width,
                    bboxes: cropped_bboxes,
                };

                (cropped_record, cropped_image)
            },
        );

    let (record_tl, cropped_tl) = record_cropped_iter.next().await.unwrap();
    let (record_tr, cropped_tr) = record_cropped_iter.next().await.unwrap();
    let (record_bl, cropped_bl) = record_cropped_iter.next().await.unwrap();
    let (record_br, cropped_br) = record_cropped_iter.next().await.unwrap();
    debug_assert_eq!(record_cropped_iter.next().await, None);

    // merge cropped images
    let merged_top = Tensor::cat(&[cropped_tl, cropped_tr], 2);
    debug_assert_eq!(merged_top.size3().unwrap(), (3, pivot_row, image_size));

    let merged_bottom = Tensor::cat(&[cropped_bl, cropped_br], 2);
    debug_assert_eq!(
        merged_bottom.size3().unwrap(),
        (3, image_size - pivot_row, image_size)
    );

    let merged_image = Tensor::cat(&[merged_top, merged_bottom], 1);
    debug_assert_eq!(merged_image.size3().unwrap(), (3, image_size, image_size));

    // merge cropped records
    let merged_bboxes: Vec<_> = record_tl
        .bboxes
        .into_iter()
        .chain(record_tr.bboxes.into_iter())
        .chain(record_bl.bboxes.into_iter())
        .chain(record_br.bboxes.into_iter())
        .collect();

    Ok((merged_bboxes, merged_image))
}
