use super::*;
use crate::{
    common::*,
    config::Config,
    message::LoggingMessage,
    util::{PixelBBox, Ratio, RatioBBox},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Record {
    pub path: PathBuf,
    pub height: usize,
    pub width: usize,
    pub bboxes: Vec<PixelBBox>,
}

#[derive(Debug, TensorLike)]
pub struct TrainingRecord {
    pub epoch: usize,
    pub step: usize,
    pub image: Tensor,
    #[tensor_like(clone)]
    pub bboxes: Vec<Vec<RatioBBox>>,
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
                            PixelBBox::new([y.into(), x.into(), h.into(), w.into()], category_id)
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

        // load and cache images
        let stream = {
            let cache_loader =
                Arc::new(CacheLoader::new(&cache_dir, image_size as usize, 3).await?);
            let logging_tx = logging_tx.clone();

            stream.and_then(move |args| {
                let (step, epoch, record_vec) = args;
                let cache_loader = cache_loader.clone();
                let logging_tx = logging_tx.clone();

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

                    Fallible::Ok((step, epoch, bbox_image_vec))
                }
            })
        };

        // start of unordered ops
        let stream = Box::pin(stream).try_overflowing_enumerate();

        // make mosaic
        let stream = {
            let mosaic_processor = Arc::new(MosaicProcessor::new(image_size, mosaic_margin));

            stream.try_par_then_unordered(None, move |(index, args)| {
                let mosaic_processor = mosaic_processor.clone();
                let mut rng = StdRng::from_entropy();
                let logging_tx = logging_tx.clone();

                async move {
                    let (step, epoch, bbox_image_vec) = args;

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

                    Fallible::Ok((index, (step, epoch, merged_bboxes, merged_image)))
                }
            })
        };

        // add batch dimension
        let stream = stream.try_par_then_unordered(None, move |(index, args)| async move {
            let (step, epoch, bboxes, image) = args;
            let new_image = image.unsqueeze(0);
            Fallible::Ok((index, (step, epoch, bboxes, new_image)))
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

        let stream = stream.try_par_then_unordered(None, move |(index, args)| {
            let random_affine = random_affine.clone();
            let mut rng = StdRng::from_entropy();

            async move {
                let (step, epoch, bboxes, image) = args;

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

                Fallible::Ok((index, (step, epoch, new_bboxes, new_image)))
            }
        });

        // reorder items
        let stream = Box::pin(stream).try_reorder_enumerated();

        // group into chunks
        let stream = stream
            .chunks(mini_batch_size)
            .overflowing_enumerate()
            .par_then_unordered(None, |(index, results)| async move {
                let chunk = results.into_iter().collect::<Fallible<Vec<_>>>()?;
                Fallible::Ok((index, chunk))
            });

        // convert to batched type
        let stream = stream.try_par_then_unordered(None, |(index, chunk)| {
            // summerizable type
            struct State {
                pub step: usize,
                pub epoch: usize,
                pub bboxes_vec: Vec<Vec<RatioBBox>>,
                pub image_vec: Vec<Tensor>,
            }

            impl Sum<(usize, usize, Vec<RatioBBox>, Tensor)> for State {
                fn sum<I>(mut iter: I) -> Self
                where
                    I: Iterator<Item = (usize, usize, Vec<RatioBBox>, Tensor)>,
                {
                    let (mut min_step, mut min_epoch, bboxes, image) =
                        iter.next().expect("the iterator canont be empty");
                    let mut bboxes_vec = vec![bboxes];
                    let mut image_vec = vec![image];

                    while let Some((step, epoch, bboxes, image)) = iter.next() {
                        min_step = min_step.min(step);
                        min_epoch = min_epoch.min(epoch);
                        bboxes_vec.push(bboxes);
                        image_vec.push(image);
                    }

                    Self {
                        step: min_step,
                        epoch: min_epoch,
                        bboxes_vec,
                        image_vec,
                    }
                }
            }

            async move {
                let State {
                    step,
                    epoch,
                    bboxes_vec,
                    image_vec,
                } = chunk.into_iter().sum();

                let image_batch = Tensor::stack(&image_vec, 0);

                Fallible::Ok((index, (step, epoch, bboxes_vec, image_batch)))
            }
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

        // reorder back
        let stream = stream.try_reorder_enumerated();

        let stream = Box::pin(stream);
        Ok(stream)
    }
}
