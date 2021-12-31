use crate::{common::*, config, logging::LoggingMessage};
use yolo_dl::{
    dataset::{
        CocoDataset, CsvDataset, DataRecord, FileCacheDataset, IiiDataset, MemoryCacheDataset,
        OnDemandDataset, RandomAccessDataset, SanitizedDataset, VocDataset,
    },
    processor::{MosaicProcessorInit, RandomAffineInit},
    profiling::Timing,
};

/// Asynchronous data stream for training.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct TrainingStream {
    batch_size: usize,
    #[derivative(Debug = "ignore")]
    preprocessor_config: Box<dyn Send + Borrow<config::Preprocessor>>,
    logging_tx: Option<broadcast::Sender<LoggingMessage>>,
    dataset: Arc<dyn RandomAccessDataset + Sync>,
}

impl TrainingStream {
    pub async fn new(
        batch_size: usize,
        dataset_config: impl 'static + Send + Borrow<config::Dataset>,
        preprocessor_config: impl 'static + Send + Borrow<config::Preprocessor>,
        logging_tx: Option<broadcast::Sender<LoggingMessage>>,
    ) -> Result<Self> {
        let dataset = {
            let config::Dataset {
                kind,
                class_whitelist,
                ..
            } = dataset_config.borrow();
            let config::Preprocessor {
                pipeline: config::Pipeline { device, .. },
                ref cache,
                cleanse:
                    config::Cleanse {
                        out_of_bound_tolerance,
                        min_bbox_size,
                        ..
                    },
                ..
            } = *preprocessor_config.borrow();

            match *kind {
                config::DatasetKind::Coco {
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

                    let dataset: Box<dyn RandomAccessDataset + Sync> = match cache {
                        config::Cache::NoCache => {
                            let dataset =
                                OnDemandDataset::new(dataset, image_size.get(), device).await?;
                            Box::new(dataset)
                        }
                        config::Cache::FileCache { cache_dir } => {
                            let dataset =
                                FileCacheDataset::new(dataset, cache_dir, image_size.get(), device)
                                    .await?;
                            Box::new(dataset)
                        }
                        config::Cache::MemoryCache => {
                            let dataset =
                                MemoryCacheDataset::new(dataset, image_size.get(), device).await?;
                            Box::new(dataset)
                        }
                    };

                    dataset
                }
                config::DatasetKind::Voc {
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
                    let dataset: Box<dyn RandomAccessDataset + Sync> = match cache {
                        config::Cache::NoCache => {
                            let dataset =
                                OnDemandDataset::new(dataset, image_size.get(), device).await?;
                            Box::new(dataset)
                        }
                        config::Cache::FileCache { cache_dir } => {
                            let dataset =
                                FileCacheDataset::new(dataset, cache_dir, image_size.get(), device)
                                    .await?;
                            Box::new(dataset)
                        }
                        config::Cache::MemoryCache => {
                            let dataset =
                                MemoryCacheDataset::new(dataset, image_size.get(), device).await?;
                            Box::new(dataset)
                        }
                    };
                    dataset
                }
                config::DatasetKind::Iii {
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
                    let dataset: Box<dyn RandomAccessDataset + Sync> = match cache {
                        config::Cache::NoCache => {
                            let dataset =
                                OnDemandDataset::new(dataset, image_size.get(), device).await?;
                            Box::new(dataset)
                        }
                        config::Cache::FileCache { cache_dir } => {
                            let dataset =
                                FileCacheDataset::new(dataset, cache_dir, image_size.get(), device)
                                    .await?;
                            Box::new(dataset)
                        }
                        config::Cache::MemoryCache => {
                            let dataset =
                                MemoryCacheDataset::new(dataset, image_size.get(), device).await?;
                            Box::new(dataset)
                        }
                    };
                    dataset
                }
                config::DatasetKind::Csv {
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
                    let dataset: Box<dyn RandomAccessDataset + Sync> = match cache {
                        config::Cache::NoCache => {
                            let dataset =
                                OnDemandDataset::new(dataset, image_size.get(), device).await?;
                            Box::new(dataset)
                        }
                        config::Cache::FileCache { cache_dir } => {
                            let dataset =
                                FileCacheDataset::new(dataset, cache_dir, image_size.get(), device)
                                    .await?;
                            Box::new(dataset)
                        }
                        config::Cache::MemoryCache => {
                            let dataset =
                                MemoryCacheDataset::new(dataset, image_size.get(), device).await?;
                            Box::new(dataset)
                        }
                    };
                    dataset
                }
            }
        };

        Ok(Self {
            batch_size,
            preprocessor_config: Box::new(preprocessor_config),
            logging_tx,
            dataset: dataset.into(),
        })
    }

    pub fn train_stream(&self) -> Result<BoxStream<'static, Result<TrainingRecord>>> {
        // parallel stream config
        let par_config: par_stream::ParParams = {
            let buf_size: par_stream::BufSize = self
                .preprocessor_config
                .deref()
                .borrow()
                .pipeline
                .worker_buf_size
                .map(|buf_size| Some(buf_size).into())
                .unwrap_or(2.0.into());

            Some(par_stream::ParParamsConfig::Manual {
                num_workers: par_stream::NumWorkers::Default,
                buf_size: buf_size,
            })
            .into()
        };

        // repeating
        let stream = stream::iter(0..);

        // sample 4 records per step
        let stream = {
            let num_records = self.dataset.num_records();

            stream.flat_map(move |epoch| {
                let mut rng = OsRng;

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
            .map(|(step, (epoch, indexes))| (step, epoch, indexes));

        // start of unordered ops
        let stream = stream.enumerate();

        // load samples and scale bboxes
        let stream = {
            let config::Preprocessor {
                mixup:
                    config::MixUp {
                        mixup_prob,
                        cutmix_prob,
                        mosaic_prob,
                        ..
                    },
                cleanse: config::Cleanse { bbox_scaling, .. },
                ..
            } = *self.preprocessor_config.deref().borrow();
            let mixup_prob = mixup_prob.to_f64();
            let cutmix_prob = cutmix_prob.to_f64();
            let mosaic_prob = mosaic_prob.to_f64();
            let dataset = self.dataset.clone();
            let logging_tx = self.logging_tx.clone();
            let par_config = par_config.clone();

            stream
                .map(Ok)
                .try_par_then_unordered(par_config.clone(), move |args| {
                    let (index, (step, epoch, record_indexes)) = args;
                    let dataset = dataset.clone();
                    let logging_tx = logging_tx.clone();
                    let mut timing = Timing::new("pipeline");
                    let mut rng = StdRng::from_entropy();
                    let par_config = par_config.clone();

                    async move {
                        timing.add_event("init");

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
                        let image_bbox_vec: Vec<_> = stream::iter(record_indexes)
                            .take(mix_kind.num_samples())
                            .par_then_unordered(par_config, move |record_index| {
                                let dataset = dataset.clone();

                                async move {
                                    // load sample
                                    let DataRecord { image, bboxes } =
                                        dataset.nth(record_index).await?;

                                    // scale bboxes
                                    let bboxes: Vec<_> = bboxes
                                        .into_iter()
                                        .map(|bbox| RatioRectLabel {
                                            rect: bbox.rect.scale_size(bbox_scaling).unwrap(),
                                            class: bbox.class,
                                        })
                                        .collect();
                                    Fallible::Ok((image, bboxes))
                                }
                                .instrument(trace_span!("load_sample"))
                            })
                            .try_collect()
                            .await?;

                        // pack into mixed data type
                        let data = MixData::new(mix_kind, image_bbox_vec).unwrap();

                        // send to logger
                        if let Some(logging_tx) = &logging_tx {
                            let (images, bboxes) = data
                                .as_ref()
                                .iter()
                                .map(|(image, bboxes)| (image.shallow_clone(), bboxes.clone()))
                                .unzip_n_vec();

                            let msg = LoggingMessage::new_debug_images(
                                "sample-loading",
                                images,
                                Some(bboxes),
                            );
                            let _ = logging_tx.send(msg);
                        }

                        timing.add_event("data loading");

                        Fallible::Ok((index, (step, epoch, data, timing)))
                    }
                    .instrument(trace_span!("data_loading"))
                })
        };

        // color jitter
        let stream = {
            let color_jitter_config = &self.preprocessor_config.deref().borrow().color_jitter;
            let color_jitter_prob = color_jitter_config.color_jitter_prob;
            let color_jitter = Arc::new(color_jitter_config.color_jitter_init().build());
            let logging_tx = self.logging_tx.clone();
            let par_config = par_config.clone();

            stream.try_par_map_unordered(par_config.clone(), move |(index, args)| {
                let color_jitter = color_jitter.clone();
                let logging_tx = logging_tx.clone();

                move || {
                    let (step, epoch, input, mut timing) = args;
                    timing.add_event("in channel");

                    let mix_kind = input.kind();
                    let pairs: Vec<_> = input
                        .into_iter()
                        .map(|(image, bboxes)| -> Result<_> {
                            let mut rng = OsRng;
                            let yes = rng.gen_bool(color_jitter_prob.to_f64());

                            let (new_image, new_bboxes) = if yes {
                                let new_image = color_jitter.forward(&image)?;
                                (new_image, bboxes)
                            } else {
                                (image, bboxes)
                            };
                            Ok((new_image, new_bboxes))
                        })
                        .try_collect()?;
                    let output = MixData::new(mix_kind, pairs).unwrap();

                    // send to logger
                    if let Some(logging_tx) = &logging_tx {
                        let (images, bboxes) = output
                            .iter()
                            .map(|(image, bboxes)| (image, bboxes))
                            .unzip_n_vec();

                        let msg =
                            LoggingMessage::new_debug_images("color-jitter", images, Some(bboxes));
                        let _result = logging_tx.send(msg);
                    }

                    timing.add_event("color jitter");
                    Fallible::Ok((index, (step, epoch, output, timing)))
                }
            })
        };

        // random affine
        let stream = {
            let config::Preprocessor {
                random_affine:
                    config::RandomAffine {
                        affine_prob,
                        rotate_prob,
                        rotate_degrees,
                        translation_prob,
                        translation,
                        scale_prob,
                        scale,
                        horizontal_flip_prob,
                        vertical_flip_prob,
                    },
                cleanse:
                    config::Cleanse {
                        min_bbox_size,
                        min_bbox_cropping_ratio,
                        ..
                    },
                ..
            } = *self.preprocessor_config.deref().borrow();
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
                    min_bbox_size: Some(R64::from(min_bbox_size)),
                    min_bbox_cropping_ratio: Some(min_bbox_cropping_ratio.into()),
                }
                .build()?,
            );
            let logging_tx = self.logging_tx.clone();
            let par_config = par_config.clone();

            stream.try_par_map_unordered(par_config.clone(), move |(index, args)| {
                let random_affine = random_affine.clone();
                let logging_tx = logging_tx.clone();

                move || {
                    let (step, epoch, data, mut timing) = args;
                    timing.add_event("in channel");

                    let mix_kind = data.kind();
                    let pairs: Vec<_> = data
                        .into_iter()
                        .map(move |(image, bboxes)| -> Result<_> {
                            let mut rng = OsRng;

                            let yes = rng.gen_bool(affine_prob.to_f64());
                            let (new_image, new_bboxes) = if yes {
                                random_affine.forward(&image, &bboxes)?
                            } else {
                                (image, bboxes)
                            };
                            Ok((new_image, new_bboxes))
                        })
                        .try_collect()?;

                    let new_data = MixData::new(mix_kind, pairs).unwrap();

                    // send to logger
                    if let Some(logging_tx) = &logging_tx {
                        let (images, bboxes) = new_data
                            .iter()
                            .map(|(image, bboxes)| (image, bboxes))
                            .unzip_n_vec();

                        let msg =
                            LoggingMessage::new_debug_images("random-affine", images, Some(bboxes));
                        let _result = logging_tx.send(msg);
                    }

                    timing.add_event("random affine");
                    Fallible::Ok((index, (step, epoch, new_data, timing)))
                }
            })
        };

        // mixup
        let stream = {
            let config::Preprocessor {
                mixup:
                    config::MixUp {
                        mixup_prob,
                        cutmix_prob,
                        mosaic_prob,
                        mosaic_margin,
                        ..
                    },
                cleanse:
                    config::Cleanse {
                        min_bbox_size,
                        min_bbox_cropping_ratio,
                        ..
                    },
                ..
            } = *self.preprocessor_config.deref().borrow();
            let mixup_prob = mixup_prob.to_f64();
            let cutmix_prob = cutmix_prob.to_f64();
            let mosaic_prob = mosaic_prob.to_f64();
            ensure!(
                mixup_prob + cutmix_prob + mosaic_prob <= 1.0 + f64::default_epsilon(),
                "the sum of mixup, cutmix, mosaic probabilities must not exceed 1.0"
            );
            let mosaic_processor = Arc::new(
                MosaicProcessorInit {
                    mosaic_margin: mosaic_margin.to_f64(),
                    min_bbox_size: Some(min_bbox_size.into()),
                    min_bbox_cropping_ratio: Some(min_bbox_cropping_ratio.into()),
                }
                .build()?,
            );
            let logging_tx = self.logging_tx.clone();

            stream.try_par_map_unordered(par_config.clone(), move |(index, args)| {
                let mosaic_processor = mosaic_processor.clone();
                let logging_tx = logging_tx.clone();

                move || {
                    let (step, epoch, data, mut timing) = args;
                    timing.add_event("in channel");

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
                        MixData::Mosaic(pairs) => mosaic_processor.forward(Vec::from(pairs))?,
                    };

                    // filter out small bboxes after mixing
                    let mixed_bboxes: Vec<_> = mixed_bboxes
                        .into_iter()
                        .filter_map(|bbox| {
                            let ok = bbox.h() >= min_bbox_size.to_r64()
                                && bbox.w() >= min_bbox_size.to_r64();
                            ok.then(|| bbox)
                        })
                        .collect();

                    // send to logger
                    if let Some(logging_tx) = &logging_tx {
                        let msg = LoggingMessage::new_debug_images(
                            "mosaic-processor",
                            vec![&mixed_image],
                            Some(vec![&mixed_bboxes]),
                        );
                        let _result = logging_tx.send(msg);
                    }

                    timing.add_event("mosaic processor");
                    Fallible::Ok((index, (step, epoch, mixed_bboxes, mixed_image, timing)))
                }
            })
        };

        // add batch dimension
        let stream = stream.try_par_map_unordered(par_config.clone(), move |(index, args)| {
            move || {
                let (step, epoch, bboxes, image, mut timing) = args;
                timing.add_event("in channel");
                let new_image = image.unsqueeze(0);
                timing.add_event("batch dimensions");
                Fallible::Ok((index, (step, epoch, bboxes, new_image, timing)))
            }
        });

        // optionally reorder records
        let stream: Pin<Box<dyn Stream<Item = Result<_>> + Send>> = {
            if self
                .preprocessor_config
                .deref()
                .borrow()
                .pipeline
                .unordered_records
            {
                stream.map_ok(|(_index, args)| args).boxed()
            } else {
                stream.try_reorder_enumerated().boxed()
            }
        };

        // group into chunks
        let stream = stream
            .chunks(self.batch_size)
            .enumerate()
            .par_map_unordered(par_config.clone(), |(index, results)| {
                move || {
                    let chunk: Vec<_> = results.into_iter().try_collect()?;
                    Fallible::Ok((index, chunk))
                }
            });

        // convert to batched type
        let stream = stream.try_par_map_unordered(par_config.clone(), |(index, mut chunk)| {
            move || {
                chunk.iter_mut().for_each(|args| {
                    let (_step, _epoch, _bboxes, _image, timing) = args;
                    timing.add_event("in channel");
                });

                let (min_step, min_epoch, bboxes_vec, image_vec, timing_vec): (
                    MinVal<_>,
                    MinVal<_>,
                    Vec<_>,
                    Vec<_>,
                    Vec<_>,
                ) = chunk.into_iter().unzip_n();
                let min_step = min_step.unwrap();
                let min_epoch = min_epoch.unwrap();
                let image_batch = Tensor::cat(&image_vec, 0);
                let timing = Timing::merge("batching", timing_vec).unwrap();

                Fallible::Ok((
                    index,
                    (min_step, min_epoch, bboxes_vec, image_batch, timing),
                ))
            }
        });

        // map to output type
        let stream = stream.map(move |result| {
            let (index, args) = result?;
            let (step, epoch, bboxes, image, mut timing) = args;
            timing.add_event("in channel");

            let mut record = TrainingRecord {
                epoch,
                step,
                image: image.set_requires_grad(false),
                bboxes,
                timing,
            };
            record.timing.add_event("map to output type");

            Ok((index, record))
        });

        // optionally reorder back
        let stream = if self
            .preprocessor_config
            .deref()
            .borrow()
            .pipeline
            .unordered_batches
        {
            stream.map_ok(|(_index, args)| args).boxed()
        } else {
            stream.try_reorder_enumerated().boxed()
        };

        Ok(stream)
    }

    pub fn input_channels(&self) -> usize {
        self.dataset.input_channels()
    }

    pub fn classes(&self) -> &IndexSet<String> {
        self.dataset.classes()
    }
}

/// The record that is accepted by training worker.
#[derive(Debug, TensorLike)]
pub struct TrainingRecord {
    pub epoch: usize,
    pub step: usize,
    pub image: Tensor,
    #[tensor_like(clone)]
    pub bboxes: Vec<Vec<RatioRectLabel<R64>>>,
    #[tensor_like(clone)]
    pub timing: Timing,
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
    None([(Tensor, Vec<RatioRectLabel<R64>>); 1]),
    MixUp([(Tensor, Vec<RatioRectLabel<R64>>); 2]),
    CutMix([(Tensor, Vec<RatioRectLabel<R64>>); 2]),
    Mosaic([(Tensor, Vec<RatioRectLabel<R64>>); 4]),
}

impl MixData {
    pub fn new(kind: MixKind, pairs: Vec<(Tensor, Vec<RatioRectLabel<R64>>)>) -> Result<Self> {
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

    pub fn iter(&self) -> impl Iterator<Item = &(Tensor, Vec<RatioRectLabel<R64>>)> {
        match self {
            Self::None(array) => array.iter(),
            Self::MixUp(array) => array.iter(),
            Self::CutMix(array) => array.iter(),
            Self::Mosaic(array) => array.iter(),
        }
    }

    pub fn into_iter(self) -> impl Iterator<Item = (Tensor, Vec<RatioRectLabel<R64>>)> {
        Vec::from(self).into_iter()
    }
}

impl AsRef<[(Tensor, Vec<RatioRectLabel<R64>>)]> for MixData {
    fn as_ref(&self) -> &[(Tensor, Vec<RatioRectLabel<R64>>)] {
        match self {
            Self::None(array) => array.as_ref(),
            Self::MixUp(array) => array.as_ref(),
            Self::CutMix(array) => array.as_ref(),
            Self::Mosaic(array) => array.as_ref(),
        }
    }
}

impl From<MixData> for Vec<(Tensor, Vec<RatioRectLabel<R64>>)> {
    fn from(data: MixData) -> Self {
        match data {
            MixData::None(array) => array.into(),
            MixData::MixUp(array) => array.into(),
            MixData::CutMix(array) => array.into(),
            MixData::Mosaic(array) => array.into(),
        }
    }
}
