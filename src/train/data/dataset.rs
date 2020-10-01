use super::*;
use crate::{common::*, config::Config, message::LoggingMessage, util::Timing};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Record {
    pub path: PathBuf,
    pub size: PixelSize<usize>,
    /// Bounding box in pixel units.
    pub bboxes: Vec<LabeledPixelBBox<R64>>,
}

#[derive(Debug, TensorLike)]
pub struct TrainingRecord {
    pub epoch: usize,
    pub step: usize,
    pub image: Tensor,
    #[tensor_like(clone)]
    pub bboxes: Vec<Vec<LabeledRatioBBox>>,
}

#[derive(Debug)]
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

        // build file cache

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
            ref cache_dir,
            image_size,
            mosaic_prob,
            mosaic_margin,
            affine_prob,
            mini_batch_size,
            rotate_degrees,
            translation,
            scale,
            shear,
            horizontal_flip,
            vertical_flip,
            ..
        } = *self.config;
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

        let records = Arc::new(
            annotations
                .iter()
                .map(|(_id, ann)| (ann.image_id, ann))
                .into_group_map()
                .into_iter()
                .map(|(image_id, anns)| {
                    let image = &images[&image_id];
                    let bboxes = anns
                        .into_iter()
                        .map(|ann| {
                            let [x, y, w, h] = ann.bbox.clone();
                            let category_id = ann.category_id;
                            let bbox =
                                PixelBBox::from_tlhw([y.into(), x.into(), h.into(), w.into()]);
                            LabeledPixelBBox { bbox, category_id }
                        })
                        .collect::<Vec<_>>();

                    Record {
                        path: image_dir.join(&image.file_name),
                        size: PixelSize::new(image.height, image.width),
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

            let mut index_iters = (0..4)
                .map(|_| {
                    let mut indexes = (0..num_records).collect_vec();
                    indexes.shuffle(&mut rng);
                    indexes.into_iter()
                })
                .collect_vec();

            let record_vec = (0..num_records)
                .map(|_| {
                    let timing = Timing::new();
                    let record_vec = index_iters
                        .iter_mut()
                        .map(|iter| iter.next().unwrap())
                        .map(|index| records[index].clone())
                        .collect_vec();
                    (epoch, record_vec, timing)
                })
                .collect_vec();

            futures::stream::iter(record_vec)
        });

        // add step count
        let stream = stream
            .enumerate()
            .map(|(step, (epoch, record_vec, timing))| Ok((step, epoch, record_vec, timing)));

        // start of unordered ops
        let stream = stream.try_overflowing_enumerate();

        // load and cache images
        let stream = {
            let cache_loader =
                Arc::new(CacheLoader::new(&cache_dir, image_size as usize, 3).await?);
            let logging_tx = logging_tx.clone();

            stream.try_par_then(None, move |args| {
                let (index, (step, epoch, record_vec, mut timing)) = args;
                let cache_loader = cache_loader.clone();
                let logging_tx = logging_tx.clone();
                timing.set_record("wait for cache loader");

                async move {
                    let load_cache_futs = record_vec
                        .into_iter()
                        .map(|record| {
                            let cache_loader = cache_loader.clone();

                            async move {
                                let Record {
                                    ref path,
                                    size: PixelSize { height, width, .. },
                                    ..
                                } = *record;

                                // load cache
                                let image = cache_loader.load_cache(path, height, width).await?;

                                // extend to specified size
                                let resized = resize_image(&image, image_size, image_size)?;

                                // convert to ratio bbox
                                let bboxes: Vec<_> = record
                                    .bboxes
                                    .iter()
                                    .map(|bbox| bbox.to_ratio_bbox(height, width))
                                    .collect();

                                Fallible::Ok((bboxes, resized))
                            }
                        })
                        .map(async_std::task::spawn);

                    let bbox_image_vec: Vec<(_, _)> =
                        futures::future::try_join_all(load_cache_futs).await?;

                    // send to logger
                    {
                        let images: Vec<_> =
                            bbox_image_vec.iter().map(|(_, image)| image).collect();
                        let msg = LoggingMessage::new_images("cache-loader", &images);
                        let _ = logging_tx.send(msg);
                    }

                    timing.set_record("cache loader");
                    Fallible::Ok((index, (step, epoch, bbox_image_vec, timing)))
                }
            })
        };

        // make mosaic
        let stream = {
            let mosaic_processor = Arc::new(MosaicProcessor::new(image_size, mosaic_margin));

            stream.try_par_then(None, move |(index, args)| {
                let mosaic_processor = mosaic_processor.clone();
                let mut rng = StdRng::from_entropy();
                let logging_tx = logging_tx.clone();

                async move {
                    let (step, epoch, bbox_image_vec, mut timing) = args;

                    // randomly create mosaic image
                    let (merged_bboxes, merged_image) =
                        if rng.gen_range(0.0, 1.0) <= mosaic_prob.raw() {
                            mosaic_processor.make_mosaic(bbox_image_vec).await?
                        } else {
                            bbox_image_vec.into_iter().next().unwrap()
                        };

                    // send to logger
                    {
                        let msg = LoggingMessage::new_image_with_bboxes(
                            "mosaicache-processor",
                            &merged_image,
                            &merged_bboxes,
                        );
                        let _ = logging_tx.send(msg);
                    }

                    timing.set_record("mosaic processor");
                    Fallible::Ok((index, (step, epoch, merged_bboxes, merged_image, timing)))
                }
            })
        };

        // add batch dimension
        let stream = stream.try_par_then(None, move |(index, args)| async move {
            let (step, epoch, bboxes, image, mut timing) = args;
            let new_image = image.unsqueeze(0);
            timing.set_record("batch dimensions");
            Fallible::Ok((index, (step, epoch, bboxes, new_image, timing)))
        });

        // apply random affine
        let random_affine = Arc::new(RandomAffine::new(
            rotate_degrees,
            translation,
            scale,
            shear,
            horizontal_flip,
            vertical_flip,
        ));

        warn!("TODO: random affine on bboxes is not implemented");
        let stream = stream.try_par_then(None, move |(index, args)| {
            let random_affine = random_affine.clone();
            let mut rng = StdRng::from_entropy();

            async move {
                let (step, epoch, bboxes, image, mut timing) = args;

                // randomly create mosaic image
                let (new_bboxes, new_image) = if rng.gen_range(0.0, 1.0) <= affine_prob.raw() {
                    let new_image = async_std::task::spawn_blocking(move || {
                        random_affine.batch_random_affine(&image)
                    })
                    .await;

                    // TODO: random affine on bboxes
                    let new_bboxes = bboxes;

                    (new_bboxes, new_image)
                } else {
                    (bboxes, image)
                };

                timing.set_record("random affine");
                Fallible::Ok((index, (step, epoch, new_bboxes, new_image, timing)))
            }
        });

        // reorder items
        let stream = stream.try_reorder_enumerated();

        // group into chunks
        let stream = stream
            .chunks(mini_batch_size)
            .overflowing_enumerate()
            .par_then(None, |(index, results)| async move {
                let chunk: Vec<_> = results.into_iter().try_collect()?;
                Fallible::Ok((index, chunk))
            });

        // convert to batched type
        let stream = stream.try_par_then(None, |(index, chunk)| {
            // summerizable type
            struct State {
                pub step: usize,
                pub epoch: usize,
                pub bboxes_vec: Vec<Vec<LabeledRatioBBox>>,
                pub image_vec: Vec<Tensor>,
                pub timing_vec: Vec<Timing>,
            }

            impl Sum<(usize, usize, Vec<LabeledRatioBBox>, Tensor, Timing)> for State {
                fn sum<I>(mut iter: I) -> Self
                where
                    I: Iterator<Item = (usize, usize, Vec<LabeledRatioBBox>, Tensor, Timing)>,
                {
                    let (mut min_step, mut min_epoch, bboxes, image, timing) =
                        iter.next().expect("the iterator canont be empty");
                    let mut bboxes_vec = vec![bboxes];
                    let mut image_vec = vec![image];
                    let mut timing_vec = vec![timing];

                    while let Some((step, epoch, bboxes, image, timing)) = iter.next() {
                        min_step = min_step.min(step);
                        min_epoch = min_epoch.min(epoch);
                        bboxes_vec.push(bboxes);
                        image_vec.push(image);
                        timing_vec.push(timing);
                    }

                    Self {
                        step: min_step,
                        epoch: min_epoch,
                        bboxes_vec,
                        image_vec,
                        timing_vec,
                    }
                }
            }

            async move {
                let State {
                    step,
                    epoch,
                    bboxes_vec,
                    image_vec,
                    timing_vec,
                } = chunk.into_iter().sum();

                let image_batch = Tensor::cat(&image_vec, 0);

                Fallible::Ok((index, (step, epoch, bboxes_vec, image_batch, timing_vec)))
            }
        });

        // map to output type
        let stream = stream.try_par_then(None, move |(index, args)| async move {
            let (step, epoch, bboxes, image, timing_vec) = args;

            // info!("{:#?}", timing_vec);

            let record = TrainingRecord {
                epoch,
                step,
                image,
                bboxes,
            };

            Ok((index, record))
        });

        // reorder back
        let stream = stream.try_reorder_enumerated();

        Ok(Box::pin(stream))
    }
}
