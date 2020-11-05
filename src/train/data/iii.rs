use super::*;
use crate::{
    common::*,
    config::{Config, DatasetConfig},
};

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
                    annotation,
                } = sample;

                let size = {
                    let voc_dataset::Size { width, height, .. } = annotation.size;
                    PixelSize::new(height, width)
                };

                let bboxes: Vec<_> = annotation
                    .object
                    .iter()
                    .filter_map(|obj| {
                        // filter by class list and whitelist
                        let class_name = &obj.name;
                        let class_index = self.classes.get_index_of(class_name)?;
                        if let Some(whitelist) = &self.config.dataset.class_whiltelist {
                            whitelist.get(class_name)?;
                        }
                        Some((obj, class_index))
                    })
                    .map(|(obj, class_index)| -> Result<_> {
                        let voc_dataset::BndBox {
                            xmin,
                            ymin,
                            xmax,
                            ymax,
                        } = obj.bndbox;
                        let bbox = LabeledPixelBBox {
                            bbox: PixelBBox::try_from_tlbr([ymin, xmin, ymax, xmax])?,
                            category_id: class_index,
                        };
                        Ok(bbox)
                    })
                    .try_collect()?;

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
                .par_then(None, move |xml_file| {
                    async move {
                        let xml_file = Arc::new(xml_file);

                        let xml_content = async_std::fs::read_to_string(&*xml_file)
                            .await
                            .with_context(|| {
                                format!("failed to read annotation file {}", xml_file.display())
                            })?;

                        let annotation = {
                            let xml_file = xml_file.clone();
                            async_std::task::spawn_blocking(move || -> Result<_> {
                                let annotation: voc_dataset::Annotation =
                                    serde_xml_rs::from_str(&xml_content).with_context(|| {
                                        format!(
                                            "failed to parse annotation file {}",
                                            xml_file.display()
                                        )
                                    })?;
                                Ok(annotation)
                            })
                            .await?
                        };

                        let image_file = {
                            let file_name =
                                format!("{}.jpg", xml_file.file_stem().unwrap().to_str().unwrap());
                            xml_file
                                .parent()
                                .ok_or_else(|| {
                                    format_err!("invalid xml path {}", xml_file.display())
                                })?
                                .join(file_name)
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
    pub image_file: PathBuf,
    pub annotation: voc_dataset::Annotation,
}
