use super::*;
use crate::common::*;
use iii_formosa_dataset as iii;

const III_DEPTH: usize = 3;

/// The Formosa dataset from Institute for Information Industry.
#[derive(Debug, Clone)]
pub struct IiiDataset {
    pub samples: Vec<IiiSample>,
    pub classes: IndexSet<String>,
    pub records: Vec<Arc<FileRecord>>,
}

impl GenericDataset for IiiDataset {
    fn input_channels(&self) -> usize {
        III_DEPTH
    }

    fn classes(&self) -> &IndexSet<String> {
        &self.classes
    }
}

impl FileDataset for IiiDataset {
    fn records(&self) -> &[Arc<FileRecord>] {
        &self.records
    }
}

impl IiiDataset {
    pub async fn load(
        dataset_dir: impl AsRef<Path>,
        classes_file: impl AsRef<Path>,
        class_whitelist: Option<HashSet<String>>,
        blacklist_files: HashSet<PathBuf>,
    ) -> Result<IiiDataset> {
        let dataset_dir = dataset_dir.as_ref();
        let classes_file = classes_file.as_ref();

        // load classes file
        let classes = load_classes_file(&classes_file).await?;

        // list xml files
        let xml_files = {
            let dataset_dir = dataset_dir.to_owned();
            tokio::task::spawn_blocking(move || {
                let xml_files: Vec<_> = glob::glob(&format!("{}/**/*.xml", dataset_dir.display()))?
                    .map(|result| -> Result<_> {
                        let path = result?;
                        let suffix = path.strip_prefix(&dataset_dir).unwrap();
                        if blacklist_files.contains(suffix) {
                            warn!("ignore blacklisted file '{}'", path.display());
                            Ok(None)
                        } else {
                            Ok(Some(path))
                        }
                    })
                    .filter_map(|result| result.transpose())
                    .try_collect()?;
                Fallible::Ok(xml_files)
            })
            .map(|result| Fallible::Ok(result??))
            .await?
        };

        // parse xml files
        let samples: Vec<_> = {
            stream::iter(xml_files.into_iter())
                .par_then(None, move |annotation_file| {
                    async move {
                        let xml_content = tokio::fs::read_to_string(&*annotation_file)
                            .await
                            .with_context(|| {
                                format!(
                                    "failed to read annotation file {}",
                                    annotation_file.display()
                                )
                            })?;

                        let annotation: iii::Annotation = {
                            tokio::task::spawn_blocking(move || {
                                serde_xml_rs::from_str(&xml_content)
                            })
                            .map(|result| Fallible::Ok(result??))
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
                            let image_file = annotation_file.parent().unwrap().join(file_name);
                            image_file
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

        // build records
        let classes = Arc::new(classes);
        let samples = Arc::new(samples);

        let records: Vec<_> = {
            let classes = classes.clone();
            let samples = samples.clone();

            tokio::task::spawn_blocking(move || {
                samples
                    .iter()
                    .map(|sample| -> Result<_> {
                        let IiiSample {
                            image_file,
                            annotation_file,
                            annotation,
                        } = sample;

                        let size = {
                            let iii::Size { width, height, .. } = annotation.size;
                            PixelSize::new(height, width).unwrap()
                        };

                        let bboxes: Vec<_> = annotation
                            .object
                            .iter()
                            .filter_map(|obj| {
                                // filter by class list and whitelist
                                let class_name = &obj.name;
                                let class_index = classes.get_index_of(class_name)?;
                                if let Some(whitelist) = &class_whitelist {
                                    whitelist.get(class_name)?;
                                }
                                Some((obj, class_index))
                            })
                            .filter_map(|(obj, class_index)| {
                                let iii::BndBox {
                                    xmin,
                                    ymin,
                                    xmax,
                                    ymax,
                                } = obj.bndbox;
                                let bbox = match PixelCyCxHW::from_tlbr(ymin, xmin, ymax, xmax) {
                                    Ok(bbox) => bbox,
                                    Err(_err) => {
                                        warn!(
                                            "failed to parse file '{}': invalid bbox {:?}",
                                            annotation_file.display(),
                                            [ymin, xmin, ymax, xmax]
                                        );
                                        return None;
                                    }
                                };

                                let labeled_bbox = PixelLabel {
                                    cycxhw: bbox,
                                    category_id: class_index,
                                };
                                Some(labeled_bbox)
                            })
                            .collect();

                        Ok(Arc::new(FileRecord {
                            path: image_file.clone(),
                            size,
                            bboxes,
                        }))
                    })
                    .try_collect()
            })
            .await??
        };

        let classes = Arc::try_unwrap(classes).unwrap();
        let samples = Arc::try_unwrap(samples).unwrap();

        Ok(IiiDataset {
            samples,
            classes,
            records,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IiiSample {
    pub image_file: PathBuf,
    pub annotation_file: PathBuf,
    pub annotation: iii::Annotation,
}
