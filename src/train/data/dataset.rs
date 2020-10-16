use super::*;
use crate::{
    common::*,
    config::{Config, DatasetConfig, DatasetKind, PreprocessorConfig, TrainingConfig},
    message::LoggingMessage,
    util::Timing,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CocoRecord {
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
            dataset: DatasetConfig { kind, .. },
            ..
        } = &*config;

        let dataset = match kind {
            DatasetKind::Coco {
                dataset_dir,
                dataset_name,
                ..
            } => coco::DataSet::load_async(dataset_dir, &dataset_name).await?,
            DatasetKind::Voc { dataset_dir, .. } => {
                let dataset_dir = dataset_dir.to_owned();
                let samples =
                    async_std::task::spawn_blocking(move || voc_dataset::load(dataset_dir)).await?;
                todo!();
            }
            DatasetKind::Iii { dataset_dir, .. } => {
                let xml_files: Vec<_> =
                    glob::glob(&format!("{}/**/*.xml", dataset_dir.display()))?.try_collect()?;
                let tasks = xml_files
                    .into_iter()
                    .map(|xml_file| async move {
                        let xml_content = async_std::fs::read_to_string(&xml_file)
                            .await
                            .with_context(|| {
                                format!("failed to read annotation file {}", xml_file.display())
                            })?;

                        let annotation = async_std::task::spawn_blocking(move || -> Result<_> {
                            let annotation: voc_dataset::Annotation =
                                serde_xml_rs::from_str(&xml_content).with_context(|| {
                                    format!(
                                        "failed to parse annotation file {}",
                                        xml_file.display()
                                    )
                                })?;
                            Ok(annotation)
                        })
                        .await?;

                        Fallible::Ok(annotation)
                    })
                    .map(async_std::task::spawn);

                let annotations = future::try_join_all(tasks).await?;
                todo!();
            }
        };

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
            dataset:
                DatasetConfig {
                    ref whitelist_classes,
                    image_size,
                    ..
                },
            preprocessor:
                PreprocessorConfig {
                    ref cache_dir,
                    mosaic_prob,
                    mosaic_margin,
                    affine_prob,
                    rotate_degrees,
                    translation,
                    scale,
                    shear,
                    horizontal_flip,
                    vertical_flip,
                    ..
                },
            training: TrainingConfig { batch_size, .. },
            ..
        } = *self.config;
        let batch_size = batch_size.get();
        let image_size = image_size.get() as i64;

        let coco::DataSet {
            instances,
            image_dir,
            ..
        } = &self.dataset;

        let annotations: HashMap<_, _> = instances
            .annotations
            .iter()
            .map(|ann| (ann.id, ann))
            .collect();
        let images: HashMap<_, _> = instances.images.iter().map(|img| (img.id, img)).collect();
        let categories: HashMap<_, _> = match whitelist_classes {
            Some(whitelist_classes) => instances
                .categories
                .iter()
                .filter_map(|cat| {
                    let Category { id, ref name, .. } = *cat;

                    // filter by whitelist
                    if whitelist_classes.contains(name) {
                        Some((id, name.to_owned()))
                    } else {
                        None
                    }
                })
                .collect(),
            None => instances
                .categories
                .iter()
                .map(|cat| {
                    let Category { id, ref name, .. } = *cat;
                    (id, name.to_owned())
                })
                .collect(),
        };

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
                        .filter_map(|ann| {
                            let [l, t, w, h] = ann.bbox.clone();
                            let category_id = ann.category_id;

                            // filter by whiltelist
                            if categories.contains_key(&category_id) {
                                let bbox =
                                    PixelBBox::from_tlhw([t.into(), l.into(), h.into(), w.into()]);
                                Some(LabeledPixelBBox { bbox, category_id })
                            } else {
                                None
                            }
                        })
                        .collect_vec();

                    CocoRecord {
                        path: image_dir.join(&image.file_name),
                        size: PixelSize::new(image.height, image.width),
                        bboxes,
                    }
                })
                .map(Arc::new)
                .collect_vec(),
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
                                // load cache
                                let (image, bboxes) = cache_loader.load_cache(&record).await?;
                                Fallible::Ok((image, bboxes))
                            }
                        })
                        .map(async_std::task::spawn);

                    let image_bbox_vec: Vec<(_, _)> =
                        futures::future::try_join_all(load_cache_futs).await?;

                    // send to logger
                    {
                        let msg = LoggingMessage::new_images_with_bboxes(
                            "cache-loader",
                            image_bbox_vec
                                .iter()
                                .map(|(image, bboxes)| (image.shallow_clone(), bboxes.clone()))
                                .collect_vec(),
                        );
                        let _ = logging_tx.send(msg);
                    }

                    timing.set_record("cache loader");
                    Fallible::Ok((index, (step, epoch, image_bbox_vec, timing)))
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
                    let (step, epoch, image_bbox_vec, mut timing) = args;

                    // randomly create mosaic image
                    let (merged_image, merged_bboxes) =
                        if rng.gen_range(0.0, 1.0) <= mosaic_prob.raw() {
                            mosaic_processor.make_mosaic(image_bbox_vec).await?
                        } else {
                            image_bbox_vec.into_iter().next().unwrap()
                        };

                    // send to logger
                    {
                        let msg = LoggingMessage::new_images_with_bboxes(
                            "mosaicache-processor",
                            vec![(&merged_image, &merged_bboxes)],
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
        // let random_affine = Arc::new(RandomAffine::new(
        //     rotate_degrees,
        //     translation,
        //     scale,
        //     shear,
        //     horizontal_flip,
        //     vertical_flip,
        // ));

        // warn!("TODO: random affine on bboxes is not implemented");
        // let stream = stream.try_par_then(None, move |(index, args)| {
        //     let random_affine = random_affine.clone();
        //     let mut rng = StdRng::from_entropy();

        //     async move {
        //         let (step, epoch, bboxes, image, mut timing) = args;

        //         // randomly create mosaic image
        //         let (new_bboxes, new_image) = if rng.gen_range(0.0, 1.0) <= affine_prob.raw() {
        //             let new_image = async_std::task::spawn_blocking(move || {
        //                 random_affine.batch_random_affine(&image)
        //             })
        //             .await;

        //             // TODO: random affine on bboxes
        //             let new_bboxes = bboxes;

        //             (new_bboxes, new_image)
        //         } else {
        //             (bboxes, image)
        //         };

        //         timing.set_record("random affine");
        //         Fallible::Ok((index, (step, epoch, new_bboxes, new_image, timing)))
        //     }
        // });

        // reorder items
        let stream = stream.try_reorder_enumerated();

        // group into chunks
        let stream = stream.chunks(batch_size).overflowing_enumerate().par_then(
            None,
            |(index, results)| async move {
                let chunk: Vec<_> = results.into_iter().try_collect()?;
                Fallible::Ok((index, chunk))
            },
        );

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
