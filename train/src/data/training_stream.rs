use super::*;
use crate::{
    common::*,
    config::{Config, DatasetConfig, DatasetKind, PreprocessorConfig, TrainingConfig},
    logging::LoggingMessage,
};

/// Asynchonous data stream for training.
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
                dataset:
                    DatasetConfig {
                        ref kind,
                        ref class_whitelist,
                        ..
                    },
                preprocessor:
                    PreprocessorConfig {
                        ref cache_dir,
                        out_of_bound_tolerance,
                        min_bbox_size,
                        device,
                        ..
                    },
                ..
            } = *config;

            match *kind {
                DatasetKind::Coco {
                    ref dataset_dir,
                    ref classes_file,
                    ref dataset_name,
                    image_size,
                    ..
                } => {
                    let dataset = CocoDataset::load(
                        dataset_dir,
                        classes_file,
                        class_whitelist.clone(),
                        dataset_name,
                    )
                    .await?;
                    let dataset =
                        SanitizedDataset::new(dataset, out_of_bound_tolerance, min_bbox_size)?;
                    let dataset =
                        CachedDataset::new(dataset, cache_dir, image_size.get(), device).await?;
                    let dataset: Box<dyn RandomAccessDataset> = Box::new(dataset);
                    dataset
                }
                DatasetKind::Voc {
                    ref dataset_dir,
                    ref classes_file,
                    image_size,
                    ..
                } => {
                    let dataset =
                        VocDataset::load(dataset_dir, classes_file, class_whitelist.clone())
                            .await?;
                    let dataset =
                        SanitizedDataset::new(dataset, out_of_bound_tolerance, min_bbox_size)?;
                    let dataset =
                        CachedDataset::new(dataset, cache_dir, image_size.get(), device).await?;
                    let dataset: Box<dyn RandomAccessDataset> = Box::new(dataset);
                    dataset
                }
                DatasetKind::Iii {
                    ref dataset_dir,
                    ref classes_file,
                    ref blacklist_files,
                    image_size,
                    ..
                } => {
                    let dataset = IiiDataset::load(
                        dataset_dir,
                        classes_file,
                        class_whitelist.clone(),
                        blacklist_files.clone(),
                    )
                    .await?;
                    let dataset =
                        SanitizedDataset::new(dataset, out_of_bound_tolerance, min_bbox_size)?;
                    let dataset =
                        CachedDataset::new(dataset, cache_dir, image_size.get(), device).await?;
                    let dataset: Box<dyn RandomAccessDataset> = Box::new(dataset);
                    dataset
                }
                DatasetKind::Csv {
                    ref image_dir,
                    ref label_file,
                    ref classes_file,
                    image_size,
                    input_channels,
                    ..
                } => {
                    let dataset = CsvDataset::load(
                        image_dir,
                        label_file,
                        classes_file,
                        input_channels.get(),
                        class_whitelist.clone(),
                    )
                    .await?;
                    let dataset =
                        SanitizedDataset::new(dataset, out_of_bound_tolerance, min_bbox_size)?;
                    let dataset =
                        CachedDataset::new(dataset, cache_dir, image_size.get(), device).await?;
                    let dataset: Box<dyn RandomAccessDataset> = Box::new(dataset);
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
        // parallel stream config
        let par_config: ParStreamConfig = {
            match self.config.preprocessor.worker_buf_size {
                Some(buf_size) => (1.0, buf_size).into(),
                None => (1.0, 2.0).into(),
            }
        };

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
        let stream = stream.try_wrapping_enumerate();

        // load samples and scale bboxes
        let stream = {
            let Config {
                preprocessor:
                    PreprocessorConfig {
                        bbox_scaling,
                        mixup_prob,
                        cutmix_prob,
                        mosaic_prob,
                        ..
                    },
                ..
            } = *self.config;
            let mixup_prob = mixup_prob.to_f64();
            let cutmix_prob = cutmix_prob.to_f64();
            let mosaic_prob = mosaic_prob.to_f64();
            let dataset = self.dataset.clone();
            let logging_tx = self.logging_tx.clone();
            let par_config = par_config.clone();

            stream.try_par_then_unordered(par_config.clone(), move |args| {
                let (index, (step, epoch, record_indexes)) = args;
                let dataset = dataset.clone();
                let logging_tx = logging_tx.clone();
                let mut timing = Timing::new("pipeline");
                let mut rng = StdRng::from_entropy();
                let par_config = par_config.clone();

                async move {
                    timing.add_event("data loading start");

                    // sample mix method
                    let mix_kind = [
                        (MixKind::None, 1.0 - mixup_prob - cutmix_prob - mosaic_prob),
                        (MixKind::MixUp, mixup_prob),
                        (MixKind::CutMix, cutmix_prob),
                        (MixKind::Mosaic, mosaic_prob),
                    ]
                    .choose_weighted(&mut rng, |(_kind, prob)| *prob)
                    .unwrap()
                    .0;

                    // load images
                    let image_bbox_vec: Vec<_> =
                        stream::iter(record_indexes.into_iter().take(mix_kind.num_samples()))
                            .par_then(par_config.clone(), move |record_index| {
                                let dataset = dataset.clone();

                                async move {
                                    // load sample
                                    let DataRecord { image, bboxes } =
                                        dataset.nth(record_index).await?;

                                    // scale bboxes
                                    let bboxes: Vec<_> = bboxes
                                        .into_iter()
                                        .map(|bbox| RatioLabel {
                                            cycxhw: bbox.cycxhw.scale_size(bbox_scaling).unwrap(),
                                            category_id: bbox.category_id,
                                        })
                                        .collect();
                                    Fallible::Ok((image, bboxes))
                                }
                            })
                            .try_collect()
                            .await?;

                    // pack into mixed data type
                    let data = MixData::new(mix_kind, image_bbox_vec).unwrap();

                    // send to logger
                    {
                        let msg = LoggingMessage::new_debug_labeled_images(
                            "sample-loading",
                            data.as_ref()
                                .iter()
                                .map(|(image, bboxes)| (image.shallow_clone(), bboxes.clone()))
                                .collect_vec(),
                        );
                        let _ = logging_tx.send(msg);
                    }

                    timing.add_event("data loading end");
                    Fallible::Ok((index, (step, epoch, data, timing)))
                }
            })
        };

        // random affine
        let stream = {
            let Config {
                preprocessor:
                    PreprocessorConfig {
                        affine_prob,
                        rotate_prob,
                        rotate_degrees,
                        translation_prob,
                        translation,
                        scale_prob,
                        scale,
                        horizontal_flip_prob,
                        vertical_flip_prob,
                        ..
                    },
                ..
            } = *self.config;
            let random_affine = Arc::new(
                RandomAffineInit {
                    rotate_prob,
                    rotate_radians: rotate_degrees.map(|degrees| degrees.to_radians()),
                    translation_prob,
                    translation,
                    scale_prob,
                    scale,
                    horizontal_flip_prob,
                    vertical_flip_prob,
                }
                .build()?,
            );
            let logging_tx = self.logging_tx.clone();
            let par_config = par_config.clone();

            stream.try_par_then_unordered(par_config.clone(), move |(index, args)| {
                let random_affine = random_affine.clone();
                let logging_tx = logging_tx.clone();
                let par_config = par_config.clone();

                async move {
                    let (step, epoch, data, mut timing) = args;
                    timing.add_event("random affine start");

                    let mix_kind = data.kind();
                    let pairs: Vec<_> = stream::iter(data.into_iter())
                        .par_map(par_config.clone(), move |(image, bboxes)| {
                            // TODO: fix cumbersome writing
                            let random_affine = random_affine.clone();
                            let mut rng = StdRng::from_entropy();

                            move || -> Result<_> {
                                let (new_image, new_bboxes) = if rng.gen_bool(affine_prob.to_f64())
                                {
                                    random_affine.forward(&image, &bboxes)?
                                } else {
                                    (image, bboxes)
                                };
                                Ok((new_image, new_bboxes))
                            }
                        })
                        .try_collect()
                        .await?;

                    let new_data = MixData::new(mix_kind, pairs).unwrap();

                    // send to logger
                    logging_tx
                        .send(LoggingMessage::new_debug_labeled_images(
                            "random-affine",
                            new_data
                                .iter()
                                .map(|(image, bboxes)| (image, bboxes))
                                .collect_vec(),
                        ))
                        .unwrap();

                    timing.add_event("random affine end");
                    Fallible::Ok((index, (step, epoch, new_data, timing)))
                }
            })
        };

        // mixup
        let stream = {
            let Config {
                preprocessor:
                    PreprocessorConfig {
                        mixup_prob,
                        cutmix_prob,
                        mosaic_prob,
                        mosaic_margin,
                        min_bbox_size,
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

            stream.try_par_then_unordered(par_config.clone(), move |(index, args)| {
                let mosaic_processor = mosaic_processor.clone();
                let logging_tx = logging_tx.clone();

                async move {
                    let (step, epoch, data, mut timing) = args;
                    timing.add_event("mosaic processor start");

                    let (mixed_image, mixed_bboxes) = match data {
                        MixData::None(pairs) => {
                            // take the first sample and discard others
                            Vec::from(pairs).into_iter().next().unwrap()
                        }
                        MixData::MixUp(pairs) => {
                            warn!("mixup is not implemented yet");
                            Vec::from(pairs).into_iter().next().unwrap()
                        }
                        MixData::CutMix(pairs) => {
                            warn!("cutmix is not implemented yet");
                            Vec::from(pairs).into_iter().next().unwrap()
                        }
                        MixData::Mosaic(pairs) => {
                            mosaic_processor.forward(Vec::from(pairs)).await?
                        }
                    };

                    // filter out small bboxes after mixing
                    let mixed_bboxes: Vec<_> = mixed_bboxes
                        .into_iter()
                        .filter_map(|bbox| {
                            if bbox.h() >= min_bbox_size.to_r64()
                                && bbox.w() >= min_bbox_size.to_r64()
                            {
                                Some(bbox)
                            } else {
                                None
                            }
                        })
                        .collect();

                    // send to logger
                    logging_tx
                        .send(LoggingMessage::new_debug_labeled_images(
                            "mosaic-processor",
                            vec![(&mixed_image, &mixed_bboxes)],
                        ))
                        .unwrap();

                    timing.add_event("mosaic processor end");
                    Fallible::Ok((index, (step, epoch, mixed_bboxes, mixed_image, timing)))
                }
            })
        };

        // add batch dimension
        let stream =
            stream.try_par_then_unordered(par_config.clone(), move |(index, args)| async move {
                let (step, epoch, bboxes, image, mut timing) = args;
                timing.add_event("batch dimensions start");
                let new_image = image.unsqueeze(0);
                timing.add_event("batch dimensions end");
                Fallible::Ok((index, (step, epoch, bboxes, new_image, timing)))
            });

        // optionally reorder records
        let stream: Pin<Box<dyn Stream<Item = Result<_>> + Send>> = {
            if self.config.preprocessor.unordered_records {
                Box::pin(stream.and_then(|(_index, args)| async move { Fallible::Ok(args) }))
            } else {
                Box::pin(stream.try_reorder_enumerated())
            }
        };

        // group into chunks
        let stream = {
            let Config {
                training: TrainingConfig { batch_size, .. },
                ..
            } = *self.config;

            stream
                .chunks(batch_size.get())
                .wrapping_enumerate()
                .par_then_unordered(par_config.clone(), |(index, results)| async move {
                    let chunk: Vec<_> = results.into_iter().try_collect()?;
                    Fallible::Ok((index, chunk))
                })
        };

        // convert to batched type
        let stream = stream.try_par_then_unordered(par_config.clone(), |(index, mut chunk)| {
            // summerizable type
            struct State {
                pub step: usize,
                pub epoch: usize,
                pub bboxes_vec: Vec<Vec<RatioLabel>>,
                pub image_vec: Vec<Tensor>,
                pub timing_vec: Vec<Timing>,
            }

            impl Sum<(usize, usize, Vec<RatioLabel>, Tensor, Timing)> for State {
                fn sum<I>(mut iter: I) -> Self
                where
                    I: Iterator<Item = (usize, usize, Vec<RatioLabel>, Tensor, Timing)>,
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
                chunk.iter_mut().for_each(|args| {
                    let (_step, _epoch, _bboxes, _image, timing) = args;
                    timing.add_event("batching start");
                });

                let State {
                    step,
                    epoch,
                    bboxes_vec,
                    image_vec,
                    timing_vec,
                } = chunk.into_iter().sum();

                let image_batch = Tensor::cat(&image_vec, 0);
                let timing = Timing::merge("batching end", timing_vec).unwrap();

                Fallible::Ok((index, (step, epoch, bboxes_vec, image_batch, timing)))
            }
        });

        // map to output type
        let stream =
            stream.try_par_then_unordered(par_config.clone(), move |(index, args)| async move {
                let (step, epoch, bboxes, image, mut timing) = args;
                timing.add_event("map to training record");

                let record = TrainingRecord {
                    epoch,
                    step,
                    image: image.set_requires_grad(false),
                    bboxes,
                    timing,
                };

                Ok((index, record))
            });

        // optionally reorder back
        let stream: Pin<Box<dyn Stream<Item = Result<_>> + Send>> = {
            if self.config.preprocessor.unordered_batches {
                Box::pin(stream.and_then(|(_index, args)| async move { Fallible::Ok(args) }))
            } else {
                Box::pin(stream.try_reorder_enumerated())
            }
        };

        Ok(Box::pin(stream))
    }
}

#[derive(Debug, Clone, Copy)]
enum MixKind {
    None,
    MixUp,
    CutMix,
    Mosaic,
}

impl MixKind {
    pub fn num_samples(&self) -> usize {
        match self {
            Self::None => 1,
            Self::MixUp => 2,
            Self::CutMix => 2,
            Self::Mosaic => 4,
        }
    }
}

#[derive(Debug)]
enum MixData {
    None([(Tensor, Vec<RatioLabel>); 1]),
    MixUp([(Tensor, Vec<RatioLabel>); 2]),
    CutMix([(Tensor, Vec<RatioLabel>); 2]),
    Mosaic([(Tensor, Vec<RatioLabel>); 4]),
}

impl MixData {
    pub fn new(kind: MixKind, pairs: Vec<(Tensor, Vec<RatioLabel>)>) -> Result<Self> {
        let data = (|| {
            let data = match kind {
                MixKind::None => Self::None(pairs.try_into()?),
                MixKind::MixUp => Self::MixUp(pairs.try_into()?),
                MixKind::CutMix => Self::CutMix(pairs.try_into()?),
                MixKind::Mosaic => Self::Mosaic(pairs.try_into()?),
            };
            Ok(data)
        })()
        .map_err(|pairs: Vec<_>| {
            format_err!(
                "expect {} pairs, but get {}",
                kind.num_samples(),
                pairs.len()
            )
        })?;
        Ok(data)
    }

    pub fn kind(&self) -> MixKind {
        match self {
            Self::None(_) => MixKind::None,
            Self::MixUp(_) => MixKind::MixUp,
            Self::CutMix(_) => MixKind::CutMix,
            Self::Mosaic(_) => MixKind::Mosaic,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &(Tensor, Vec<RatioLabel>)> {
        match self {
            Self::None(array) => array.iter(),
            Self::MixUp(array) => array.iter(),
            Self::CutMix(array) => array.iter(),
            Self::Mosaic(array) => array.iter(),
        }
    }

    pub fn into_iter(self) -> impl Iterator<Item = (Tensor, Vec<RatioLabel>)> {
        Vec::from(self).into_iter()
    }
}

impl AsRef<[(Tensor, Vec<RatioLabel>)]> for MixData {
    fn as_ref(&self) -> &[(Tensor, Vec<RatioLabel>)] {
        match self {
            Self::None(array) => array.as_ref(),
            Self::MixUp(array) => array.as_ref(),
            Self::CutMix(array) => array.as_ref(),
            Self::Mosaic(array) => array.as_ref(),
        }
    }
}

impl From<MixData> for Vec<(Tensor, Vec<RatioLabel>)> {
    fn from(data: MixData) -> Self {
        match data {
            MixData::None(array) => array.into(),
            MixData::MixUp(array) => array.into(),
            MixData::CutMix(array) => array.into(),
            MixData::Mosaic(array) => array.into(),
        }
    }
}
