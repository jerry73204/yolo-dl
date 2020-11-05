use super::*;
use crate::{
    common::*,
    config::{Config, DatasetConfig},
};
use voc_dataset as voc;

const III_DEPTH: usize = 3;

#[derive(Debug, Clone)]
pub struct IiiDataset {
    pub config: Arc<Config>,
    pub samples: Vec<IiiSample>,
    pub classes: IndexSet<String>,
}

impl GenericDataset for IiiDataset {
    fn input_channels(&self) -> usize {
        III_DEPTH
    }

    fn num_classes(&self) -> usize {
        self.classes.len()
    }

    fn classes(&self) -> &IndexSet<String> {
        &self.classes
    }

    fn records(&self) -> Result<Vec<Arc<DataRecord>>> {
        let records: Vec<_> = self
            .samples
            .iter()
            .map(|sample| -> Result<_> {
                let IiiSample {
                    image_file,
                    annotation_file,
                    annotation,
                } = sample;

                let size = {
                    let voc::Size { width, height, .. } = annotation.size;
                    PixelSize::new(height, width)
                };

                let bboxes: Vec<_> = annotation
                    .object
                    .iter()
                    .filter_map(|obj| {
                        // filter by class list and whitelist
                        let class_name = &obj.name;
                        let class_index = self.classes.get_index_of(class_name)?;
                        if let Some(whitelist) = &self.config.dataset.class_whitelist {
                            whitelist.get(class_name)?;
                        }
                        Some((obj, class_index))
                    })
                    .filter_map(|(obj, class_index)| {
                        let voc::BndBox {
                            xmin,
                            ymin,
                            xmax,
                            ymax,
                        } = obj.bndbox;
                        let bbox = match PixelBBox::try_from_tlbr([ymin, xmin, ymax, xmax]) {
                            Ok(bbox) => bbox,
                            Err(_err) => {
                                warn!(
                                    "failed to parse '{}': invalid bbox {:?}",
                                    annotation_file.display(),
                                    [ymin, xmin, ymax, xmax]
                                );
                                return None;
                            }
                        };

                        let labeled_bbox = LabeledPixelBBox {
                            bbox,
                            category_id: class_index,
                        };
                        Some(labeled_bbox)
                    })
                    .collect();
                // .map(|(obj, class_index)| -> Result<_> {
                //     let voc::BndBox {
                //         xmin,
                //         ymin,
                //         xmax,
                //         ymax,
                //     } = obj.bndbox;
                //     let bbox = PixelBBox::try_from_tlbr([ymin, xmin, ymax, xmax])
                //         .with_context(|| {
                //             format!(
                //                 "failed to parse annotation file '{}'",
                //                 annotation_file.display()
                //             )
                //         })?;
                //     let labeled_bbox = LabeledPixelBBox {
                //         bbox,
                //         category_id: class_index,
                //     };
                //     Ok(labeled_bbox)
                // })
                // .try_collect()?;

                Ok(Arc::new(DataRecord {
                    path: image_file.clone(),
                    size,
                    bboxes,
                }))
            })
            .try_collect()?;

        Ok(records)
    }
}

impl IiiDataset {
    pub async fn load<P>(config: Arc<Config>, dataset_dir: P) -> Result<IiiDataset>
    where
        P: AsRef<Path>,
    {
        let Config {
            dataset: DatasetConfig { classes_file, .. },
            ..
        } = &*config;
        let dataset_dir = dataset_dir.as_ref();

        // load classes file
        let classes = load_classes_file(&classes_file).await?;

        // list xml files
        let xml_files = {
            let dataset_dir = dataset_dir.to_owned();
            async_std::task::spawn_blocking(move || {
                let xml_files: Vec<_> =
                    glob::glob(&format!("{}/**/*.xml", dataset_dir.display()))?.try_collect()?;
                Fallible::Ok(xml_files)
            })
            .await?
        };

        // parse xml files
        let samples: Vec<_> = {
            stream::iter(xml_files.into_iter())
                .par_then(None, move |annotation_file| {
                    let annotation_file = Arc::new(annotation_file);

                    async move {
                        let xml_content = async_std::fs::read_to_string(&*annotation_file)
                            .await
                            .with_context(|| {
                                format!(
                                    "failed to read annotation file {}",
                                    annotation_file.display()
                                )
                            })?;

                        let annotation: voc::Annotation = {
                            async_std::task::spawn_blocking(move || {
                                serde_xml_rs::from_str(&xml_content)
                            })
                            .await
                            .with_context(|| {
                                format!(
                                    "failed to parse annotation file {}",
                                    annotation_file.display()
                                )
                            })?
                        };

                        let image_file = {
                            let file_name = format!(
                                "{}.jpg",
                                annotation_file.file_stem().unwrap().to_str().unwrap()
                            );
                            let image_path = annotation_file.parent().unwrap().join(file_name);
                            Arc::new(image_path)
                        };

                        // sanity check
                        ensure!(
                            annotation.size.depth == III_DEPTH,
                            "expect depth to be {}, but found {}",
                            III_DEPTH,
                            annotation.size.depth
                        );

                        let sample = IiiSample {
                            annotation,
                            annotation_file,
                            image_file,
                        };

                        Fallible::Ok(sample)
                    }
                })
                .try_collect()
                .await?
        };

        Ok(IiiDataset {
            config,
            samples,
            classes,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IiiSample {
    pub image_file: Arc<PathBuf>,
    pub annotation_file: Arc<PathBuf>,
    pub annotation: voc::Annotation,
}
