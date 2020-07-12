use crate::{common::*, config::Config};

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

pub async fn train_stream(config: Arc<Config>) -> Result<(Vec<Record>, HashMap<usize, Category>)> {
    let Config {
        ref dataset_dir,
        ref dataset_name,
        ref cache_dir,
        image_size,
        mosaic_margin,
    } = *config;
    let mosaic_margin = mosaic_margin.raw();
    let image_size = image_size.get() as i64;

    let DataSet {
        instances,
        image_dir,
        ..
    } = DataSet::load_async(dataset_dir, &dataset_name).await?;

    let annotations = instances
        .annotations
        .into_iter()
        .map(|ann| (ann.id, ann))
        .collect::<HashMap<_, _>>();
    let images = instances
        .images
        .into_iter()
        .map(|img| (img.id, img))
        .collect::<HashMap<_, _>>();
    let categories = instances
        .categories
        .into_iter()
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
        .map(|(step, (epoch, record_vec))| (step, epoch, record_vec));

    // start of unordered maps
    let stream = stream.overflowing_enumerate();

    // load and cache images
    let cache_loader = Arc::new(CacheLoader::new(&cache_dir, image_size as usize, 3).await?);

    let stream = stream.par_then_unordered(None, move |(index, args)| {
        let (step, epoch, record_vec) = args;
        let cache_loader = cache_loader.clone();

        async move {
            let load_cache_futs = record_vec
                .into_iter()
                .map(|record| {
                    let cache_loader = cache_loader.clone();

                    async move {
                        let image = cache_loader.load_cache(&record.path).await?;
                        Result::<_, Error>::Ok((record, image))
                    }
                })
                .map(async_std::task::spawn);

            let record_image_vec: Vec<(_, _)> =
                futures::future::try_join_all(load_cache_futs).await?;

            Result::<_, Error>::Ok((index, (step, epoch, record_image_vec)))
        }
    });

    // make mosaic
    let stream = stream.par_then_unordered(None, move |result| async move {
        let (index, (step, epoch, record_image_vec)) = result?;
        let mut rng = StdRng::from_entropy();
        debug_assert_eq!(record_image_vec.len(), 4);

        // random select pivot point
        let pivot_row =
            (rng.gen_range(mosaic_margin, 1.0 - mosaic_margin) * image_size as f64) as i64;
        let pivot_col =
            (rng.gen_range(mosaic_margin, 1.0 - mosaic_margin) * image_size as f64) as i64;

        // crop images
        let ranges = vec![
            (0, pivot_row, 0, pivot_col),
            (0, pivot_row, pivot_col, image_size),
            (pivot_row, image_size, 0, pivot_col),
            (pivot_row, image_size, pivot_col, image_size),
        ];

        let mut record_cropped_iter =
            futures::stream::iter(record_image_vec.into_iter().zip_eq(ranges.into_iter()))
                .par_then(
                    None,
                    |((record, image), (top, bottom, left, right))| async move {
                        let Record {
                            ref path,
                            height,
                            width,
                            ref bboxes,
                        } = *record;

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
        let merged_top = Tensor::cat(&[cropped_tl, cropped_tr], 1);
        debug_assert_eq!(merged_top.size3().unwrap(), (3, pivot_row, image_size));

        let merged_bottom = Tensor::cat(&[cropped_bl, cropped_br], 1);
        debug_assert_eq!(
            merged_bottom.size3().unwrap(),
            (3, image_size - pivot_row, image_size)
        );

        let merged_image = Tensor::cat(&[merged_top, merged_bottom], 2);
        debug_assert_eq!(merged_image.size3().unwrap(), (3, image_size, image_size));

        // merge cropped records
        let merged_bboxes: Vec<_> = izip!(
            record_tl.bboxes.into_iter(),
            record_tr.bboxes.into_iter(),
            record_bl.bboxes.into_iter(),
            record_br.bboxes.into_iter(),
        )
        .collect();

        Result::<_, Error>::Ok((index, (step, epoch, merged_bboxes, merged_image)))
    });

    todo!();

    // Ok((records, categories))
}

#[derive(Debug, Clone)]
struct CacheLoader {
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

    pub async fn load_cache<P>(&self, image_path: P) -> Result<Tensor>
    where
        P: AsRef<async_std::path::Path>,
    {
        let Self {
            image_size,
            image_channels,
            ..
        } = *self;

        let image_path = image_path.as_ref().to_owned();
        let cache_path = self.cache_dir.join(format!(
            "{}-{}",
            percent_encoding::utf8_percent_encode(image_path.to_str().unwrap(), NON_ALPHANUMERIC),
            image_size
        ));
        let component_size = mem::size_of::<f32>();
        let cache_size = image_size.pow(2) * image_channels * component_size;

        let is_valid = if cache_path.is_file().await {
            let image_modified = image_path.metadata().await?.modified()?;
            let cache_meta = cache_path.metadata().await?;
            let cache_created = cache_meta.created()?;

            if cache_created > image_modified {
                cache_meta.len() == cache_size as u64
            } else {
                false
            }
        } else {
            false
        };

        if !is_valid {
            // load and resize image
            let image = async_std::task::spawn_blocking(move || -> Result<_> {
                let image = image::open(&image_path)?
                    .resize(image_size as u32, image_size as u32, FilterType::CatmullRom)
                    .to_rgb();
                Ok(image)
            })
            .await?;

            // convert to bytes
            let mut components =
                image
                    .enumerate_pixels()
                    .flat_map(|(col, row, pixel)| {
                        pixel.channels().iter().cloned().enumerate().map(
                            move |(channel, component)| {
                                let component = component as f32 / 255.0;
                                (row, col, channel, component)
                            },
                        )
                    })
                    .collect::<Vec<_>>();

            // permute the components to channel first order
            components.sort_by_cached_key(|(row, col, channel, _byte)| (*channel, *row, *col));

            // save cache
            let bytes = components
                .into_iter()
                .flat_map(|(_, _, _, component)| Vec::from(component.to_le_bytes()))
                .collect::<Vec<_>>();
            debug_assert_eq!(bytes.len(), cache_size);
            async_std::fs::write(&cache_path, &bytes).await?;
        };

        // load from cache
        let image = Tensor::f_from_file(
            cache_path.to_str().unwrap(),
            false,
            Some(cache_size as i64),
            FLOAT_CPU,
        )?;

        Ok(image)
    }
}

fn crop_image(image: &Tensor, top: i64, bottom: i64, left: i64, right: i64) -> Tensor {
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

fn crop_bbox(bbox: &BBox, top: i64, bottom: i64, left: i64, right: i64) -> Option<BBox> {
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
