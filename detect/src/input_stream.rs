use crate::{
    common::*,
    config::{Config, DatasetKind, InputConfig, PreprocessConfig},
};

#[derive(Debug, TensorLike)]
pub struct InputRecord {
    pub indexes: Vec<usize>,
    pub images: Tensor,
    #[tensor_like(clone)]
    pub bboxes: Vec<Vec<RatioRectLabel<R64>>>,
}

#[derive(Debug)]
pub struct InputStream {
    config: Arc<Config>,
    dataset: Arc<Box<dyn StreamingDataset + Sync>>,
}

impl InputStream {
    pub async fn new(config: Arc<Config>) -> Result<Self> {
        let dataset = {
            let Config {
                input:
                    InputConfig {
                        ref kind,
                        ref class_whitelist,
                        ..
                    },
                preprocess:
                    PreprocessConfig {
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
                    let dataset = OnDemandDataset::new(dataset, image_size.get(), device).await?;
                    let dataset = RandomAccessStream::new(dataset);
                    let dataset: Box<dyn StreamingDataset + Sync> = Box::new(dataset);
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
                    let dataset = OnDemandDataset::new(dataset, image_size.get(), device).await?;
                    let dataset = RandomAccessStream::new(dataset);
                    let dataset: Box<dyn StreamingDataset + Sync> = Box::new(dataset);
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
                    let dataset = OnDemandDataset::new(dataset, image_size.get(), device).await?;
                    let dataset = RandomAccessStream::new(dataset);
                    let dataset: Box<dyn StreamingDataset + Sync> = Box::new(dataset);
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
                    let dataset = OnDemandDataset::new(dataset, image_size.get(), device).await?;
                    let dataset = RandomAccessStream::new(dataset);
                    let dataset: Box<dyn StreamingDataset + Sync> = Box::new(dataset);
                    dataset
                }
            }
        };

        Ok(Self {
            config,
            dataset: Arc::new(dataset),
        })
    }

    pub fn stream(&self) -> Result<Pin<Box<dyn Stream<Item = Result<InputRecord>> + Send>>> {
        let stream = self.dataset.stream()?;

        // add indexe
        let stream = stream.try_enumerate();

        // group into chunks
        let stream = {
            let minibatch_size = self.config.model.minibatch_size.get();

            stream
                .chunks(minibatch_size)
                .par_map_unordered(None, |results| {
                    move || {
                        let chunk: Vec<_> = results.into_iter().try_collect()?;
                        anyhow::Ok(chunk)
                    }
                })
        };

        let stream = stream.try_par_map_unordered(None, |chunk| {
            move || {
                let (indexes, images, bboxes) = chunk
                    .into_iter()
                    .map(|(index, DataRecord { image, bboxes })| (index, image, bboxes))
                    .unzip_n_vec();

                let images = Tensor::stack(&images, 0);

                Ok(InputRecord {
                    indexes,
                    images,
                    bboxes,
                })
            }
        });

        Ok(Box::pin(stream))
    }
}
