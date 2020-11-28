use super::*;
use crate::{
    common::*,
    config::{Config, DatasetConfig, DatasetKind, PreprocessorConfig, TrainingConfig},
    message::LoggingMessage,
};

#[derive(Debug)]
pub struct TrainingStream {
    config: Arc<Config>,
    logging_tx: broadcast::Sender<LoggingMessage>,
    dataset: Arc<Box<dyn RandomAccessDataset>>,
}

impl GenericDataset for TrainingStream {
    fn input_channels(&self) -> usize {
        self.dataset.input_channels()
    }

    fn classes(&self) -> &IndexSet<String> {
        self.dataset.classes()
    }
}

impl TrainingStream {
    pub async fn new(
        config: Arc<Config>,
        logging_tx: broadcast::Sender<LoggingMessage>,
    ) -> Result<Self> {
        let dataset = {
            let Config {
                dataset: DatasetConfig { ref kind, .. },
                preprocessor:
                    PreprocessorConfig {
                        ref cache_dir,
                        out_of_bound_tolerance,
                        device,
                        ..
                    },
                ..
            } = *config;

            match kind {
                DatasetKind::Coco {
                    dataset_dir,
                    dataset_name,
                    image_size,
                    ..
                } => {
                    let dataset =
                        CocoDataset::load(config.clone(), dataset_dir, dataset_name).await?;
                    let dataset = SanitizedDataset::new(dataset, out_of_bound_tolerance)?;
                    let dataset =
                        CachedDataset::new(dataset, cache_dir, image_size.get(), device).await?;
                    let dataset: Box<dyn RandomAccessDataset> = Box::new(dataset);
                    dataset
                }
                DatasetKind::Voc {
                    dataset_dir,
                    image_size,
                    ..
                } => {
                    let dataset = VocDataset::load(config.clone(), dataset_dir).await?;
                    let dataset = SanitizedDataset::new(dataset, out_of_bound_tolerance)?;
                    let dataset =
                        CachedDataset::new(dataset, cache_dir, image_size.get(), device).await?;
                    let dataset: Box<dyn RandomAccessDataset> = Box::new(dataset);
                    dataset
                }
                DatasetKind::Iii {
                    dataset_dir,
                    blacklist_files,
                    image_size,
                    ..
                } => {
                    let dataset =
                        IiiDataset::load(config.clone(), dataset_dir, blacklist_files.clone())
                            .await?;
                    let dataset = SanitizedDataset::new(dataset, out_of_bound_tolerance)?;
                    let dataset =
                        CachedDataset::new(dataset, cache_dir, image_size.get(), device).await?;
                    let dataset: Box<dyn RandomAccessDataset> = Box::new(dataset);
                    dataset
                }
                DatasetKind::Mmap { dataset_file, .. } => {
                    let dataset: Box<dyn RandomAccessDataset> =
                        Box::new(MmapDataset::load(dataset_file, device).await?);
                    dataset
                }
            }
        };

        Ok(Self {
            config,
            logging_tx,
            dataset: Arc::new(dataset),
        })
    }

    pub async fn train_stream(
        &self,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<TrainingRecord>> + Send>>> {
        // repeating
        let stream = stream::repeat(()).enumerate();

        // sample 4 records per step
        let stream = {
            let num_records = self.dataset.num_records();

            stream.flat_map(move |(epoch, ())| {
                let mut rng = rand::thread_rng();

                let mut index_iters = (0..4)
                    .map(|_| {
                        let mut indexes = (0..num_records).collect_vec();
                        indexes.shuffle(&mut rng);
                        indexes.into_iter()
                    })
                    .collect_vec();

                let indexes_vec = (0..num_records)
                    .map(|_| {
                        let indexes = index_iters
                            .iter_mut()
                            .map(|iter| iter.next().unwrap())
                            .collect_vec();
                        (epoch, indexes)
                    })
                    .collect_vec();

                futures::stream::iter(indexes_vec)
            })
        };

        // add step count
        let stream = stream
            .enumerate()
            .map(|(step, (epoch, indexes))| Ok((step, epoch, indexes)));

        // start of unordered ops
        let stream = stream.try_overflowing_enumerate();

        // load samples and scale bboxes
        let stream = {
            let Config {
                preprocessor: PreprocessorConfig { bbox_scaling, .. },
                ..
            } = *self.config;
            let dataset = self.dataset.clone();
            let logging_tx = self.logging_tx.clone();

            stream.try_par_then(None, move |args| {
                let (index, (step, epoch, record_indexes)) = args;
                let dataset = dataset.clone();
                let logging_tx = logging_tx.clone();
                let mut timing = Timing::new("pipeline");

                async move {
                    timing.set_record("data loading start");

                    let image_bbox_vec: Vec<_> = stream::iter(record_indexes.into_iter())
                        .par_then(None, move |record_index| {
                            let dataset = dataset.clone();

                            async move {
                                // laod sample
                                let DataRecord { image, bboxes } =
                                    dataset.nth(record_index).await?;

                                // scale bboxes
                                let bboxes: Vec<_> = bboxes
                                    .into_iter()
                                    .map(|bbox| bbox.scale(bbox_scaling))
                                    .collect();
                                Fallible::Ok((image, bboxes))
                            }
                        })
                        .try_collect()
                        .await?;

                    // send to logger
                    {
                        let msg = LoggingMessage::new_images_with_bboxes(
                            "sample-loading",
                            image_bbox_vec
                                .iter()
                                .map(|(image, bboxes)| (image.shallow_clone(), bboxes.clone()))
                                .collect_vec(),
                        );
                        let _ = logging_tx.send(msg);
                    }

                    timing.set_record("data loading end");
                    Fallible::Ok((index, (step, epoch, image_bbox_vec, timing)))
                }
            })
        };

        // random affine
        warn!("TODO: random affine on bboxes is not implemented");
        let stream = {
            let Config {
                preprocessor:
                    PreprocessorConfig {
                        affine_prob,
                        rotate_degrees,
                        translation,
                        scale,
                        shear,
                        horizontal_flip,
                        vertical_flip,
                        ..
                    },
                ..
            } = *self.config;

            let random_affine = Arc::new(
                RandomAffineInit {
                    rotate_radians: rotate_degrees.map(|degrees| degrees.to_radians()),
                    translation,
                    scale,
                    shear,
                    horizontal_flip,
                    vertical_flip,
                }
                .build()?,
            );

            stream.try_par_then(None, move |(index, args)| {
                let random_affine = random_affine.clone();
                let mut rng = StdRng::from_entropy();

                async move {
                    let (step, epoch, image_bbox_vec, mut timing) = args;
                    // let image_bbox_vec = image_bbox_vec.into_iter().map();
                    Fallible::Ok((index, (step, epoch, image_bbox_vec, timing)))
                }
            })
        };

        // mixup
        let stream = {
            #[derive(Debug, Clone, Copy)]
            enum MixKind {
                None,
                MixUp,
                CutMix,
                Mosaic,
            }

            let Config {
                preprocessor:
                    PreprocessorConfig {
                        mixup_prob,
                        cutmix_prob,
                        mosaic_prob,
                        mosaic_margin,
                        ..
                    },
                ..
            } = *self.config;
            let mixup_prob = mixup_prob.to_f64();
            let cutmix_prob = cutmix_prob.to_f64();
            let mosaic_prob = mosaic_prob.to_f64();
            ensure!(
                mixup_prob + cutmix_prob + mosaic_prob <= 1.0 + f64::default_epsilon(),
                "the sum of mixup, cutmix, mosaic probabilities must not exceed 1.0"
            );
            let mosaic_processor = Arc::new(
                ParallelMosaicProcessorInit {
                    mosaic_margin: mosaic_margin.to_f64(),
                    max_workers: None,
                }
                .build()?,
            );
            let logging_tx = self.logging_tx.clone();

            stream.try_par_then_unordered(None, move |(index, args)| {
                let mosaic_processor = mosaic_processor.clone();
                let mut rng = StdRng::from_entropy();
                let logging_tx = logging_tx.clone();

                async move {
                    let (step, epoch, image_bbox_vec, mut timing) = args;
                    timing.set_record("mosaic processor start");

                    let mix_kind = [
                        (MixKind::None, 1.0 - mixup_prob - cutmix_prob - mosaic_prob),
                        (MixKind::MixUp, mixup_prob),
                        (MixKind::CutMix, cutmix_prob),
                        (MixKind::Mosaic, mosaic_prob),
                    ]
                    .choose_weighted(&mut rng, |(_kind, prob)| *prob)
                    .unwrap()
                    .0;

                    let (mixed_image, mixed_bboxes) = match mix_kind {
                        MixKind::None => {
                            // take the first sample and discard others
                            image_bbox_vec.into_iter().next().unwrap()
                        }
                        MixKind::MixUp => {
                            warn!("mixup is not implemented yet");
                            image_bbox_vec.into_iter().next().unwrap()
                        }
                        MixKind::CutMix => {
                            warn!("cutmix is not implemented yet");
                            image_bbox_vec.into_iter().next().unwrap()
                        }
                        MixKind::Mosaic => mosaic_processor.forward(image_bbox_vec).await?,
                    };

                    // send to logger
                    logging_tx
                        .send(LoggingMessage::new_images_with_bboxes(
                            "mosaic-processor",
                            vec![(&mixed_image, &mixed_bboxes)],
                        ))
                        .unwrap();

                    timing.set_record("mosaic processor end");
                    Fallible::Ok((index, (step, epoch, mixed_bboxes, mixed_image, timing)))
                }
            })
        };

        // add batch dimension
        let stream = stream.try_par_then(None, move |(index, args)| async move {
            let (step, epoch, bboxes, image, mut timing) = args;
            timing.set_record("batch dimensions start");
            let new_image = image.unsqueeze(0);
            timing.set_record("batch dimensions end");
            Fallible::Ok((index, (step, epoch, bboxes, new_image, timing)))
        });

        // reorder items
        let stream = stream.try_reorder_enumerated();

        // group into chunks
        let stream = {
            let Config {
                training: TrainingConfig { batch_size, .. },
                ..
            } = *self.config;

            stream
                .chunks(batch_size.get())
                .overflowing_enumerate()
                .par_then(None, |(index, results)| async move {
                    let chunk: Vec<_> = results.into_iter().try_collect()?;
                    Fallible::Ok((index, chunk))
                })
        };

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

            timing_vec[0].report();

            let record = TrainingRecord {
                epoch,
                step,
                image: image.set_requires_grad(false),
                bboxes,
            };

            Ok((index, record))
        });

        // reorder back
        let stream = stream.try_reorder_enumerated();

        Ok(Box::pin(stream))
    }
}

pub async fn load_classes_file<P>(path: P) -> Result<IndexSet<String>>
where
    P: AsRef<async_std::path::Path>,
{
    let path = path.as_ref();
    let content = async_std::fs::read_to_string(path).await?;
    let lines: Vec<_> = content.lines().collect();
    let classes: IndexSet<_> = lines.iter().cloned().map(ToOwned::to_owned).collect();
    ensure!(
        lines.len() == classes.len(),
        "duplicated class names found in '{}'",
        path.display()
    );
    ensure!(
        classes.len() > 0,
        "no classes found in '{}'",
        path.display()
    );
    Ok(classes)
}
