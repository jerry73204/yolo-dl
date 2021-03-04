use super::*;
use crate::common::*;

/// The Pascal VOC dataset.
#[derive(Debug, Clone)]
pub struct CsvDataset {
    pub classes: IndexSet<String>,
    pub samples: Vec<CsvSample>,
    pub records: Vec<Arc<FileRecord>>,
    pub input_channels: usize,
}

impl GenericDataset for CsvDataset {
    fn input_channels(&self) -> usize {
        self.input_channels
    }

    fn classes(&self) -> &IndexSet<String> {
        &self.classes
    }
}

impl FileDataset for CsvDataset {
    fn records(&self) -> &[Arc<FileRecord>] {
        &self.records
    }
}

impl CsvDataset {
    pub async fn load(
        image_dir: impl AsRef<Path>,
        label_file: impl AsRef<Path>,
        classes_file: impl AsRef<Path>,
        input_channels: usize,
        class_whitelist: Option<HashSet<String>>,
    ) -> Result<Self> {
        let image_dir = image_dir.as_ref();
        let label_file = label_file.as_ref();
        let classes_file = classes_file.as_ref();

        // load classes file
        let classes = load_classes_file(classes_file).await?;

        // build records
        let samples = {
            let image_dir = image_dir.to_owned();
            let label_file = label_file.to_owned();
            tokio::task::spawn_blocking(move || load_csv_dataset(image_dir, label_file)).await??
        };

        // transform record type
        let classes = Arc::new(classes);
        let samples = ArcRef::new(Arc::new(samples));
        let class_whitelist = Arc::new(class_whitelist);

        let records: Vec<_> = {
            let classes = classes.clone();
            let samples = samples.clone();

            let records = tokio::task::spawn_blocking(move || {
                let n_samples = samples.len();
                (0..n_samples)
                    .map(|index| {
                        let record = samples.clone().map(|samples| &samples[index]);
                        let image_file = record.clone().map(|record| &record.image_file);
                        (image_file, record)
                    })
                    .into_group_map()
            })
            .await?;

            stream::iter(records)
                .par_map(None, move |(image_file, records)| {
                    let classes = classes.clone();
                    let class_whitelist = class_whitelist.clone();

                    move || -> Result<_> {
                        let size = {
                            let imagesize::ImageSize {
                                height: img_h,
                                width: img_w,
                            } = imagesize::size(&*image_file)?;
                            PixelSize::new(img_h, img_w)?
                        };

                        let bboxes = records
                            .into_iter()
                            .filter_map(|record| {
                                let class_name = &record.class_name;
                                if let Some(whitelist) = &*class_whitelist {
                                    whitelist.get(class_name)?;
                                }
                                let class_index = classes.get_index_of(class_name.as_str())?;
                                Some((record, class_index))
                            })
                            .map(|(record, class_index)| -> Result<_> {
                                let CsvSample { cy, cx, h, w, .. } = *record;
                                let label = PixelLabel {
                                    cycxhw: PixelCyCxHW::from_cycxhw(cy, cx, h, w)?,
                                    class: class_index,
                                };
                                Ok(label)
                            })
                            .try_collect()?;

                        let record = FileRecord {
                            path: (*image_file).to_owned(),
                            size,
                            bboxes,
                        };

                        Ok(Arc::new(record))
                    }
                })
                .try_collect()
                .await?
        };
        let classes = Arc::try_unwrap(classes).unwrap();
        let samples = Arc::try_unwrap(samples.into_owner()).unwrap();

        Ok(Self {
            classes,
            samples,
            records,
            input_channels,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize)]
pub struct CsvSample {
    pub image_file: PathBuf,
    pub class_name: String,
    pub cy: R64,
    pub cx: R64,
    pub h: R64,
    pub w: R64,
}

pub fn load_csv_dataset(
    image_dir: impl AsRef<Path>,
    label_file: impl AsRef<Path>,
) -> Result<Vec<CsvSample>> {
    let image_dir = image_dir.as_ref();
    let label_file = label_file.as_ref();

    // parse label file
    let records: Vec<CsvSample> = ::csv::ReaderBuilder::new()
        .has_headers(true)
        .comment(Some(b'#'))
        .from_path(label_file)?
        .deserialize()
        .try_collect()?;

    // check existence of image files
    let records: Vec<_> = records
        .into_iter()
        .map(|record| {
            let image_file = image_dir.join(&record.image_file);
            ensure!(
                image_file.is_file(),
                "the image file '{}' does not exist",
                image_file.display()
            );
            Ok(CsvSample {
                image_file,
                ..record
            })
        })
        .try_collect()?;

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn csv_dataset_test() {
        let base_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("csv_dataset");
        let image_dir = base_dir.join("images");
        let label_file = base_dir.join("label.csv");
        let classes_file = base_dir.join("classes.txt");

        let dataset = CsvDataset::load(image_dir, label_file, classes_file, 3, None)
            .await
            .unwrap();

        assert_eq!(dataset.records.len(), 3);
        assert_eq!(dataset.classes.len(), 3);
        assert_eq!(dataset.input_channels(), 3);
    }
}
