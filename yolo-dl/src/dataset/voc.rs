use super::*;
use crate::common::*;
use bbox::{prelude::*, CyCxHW, HW};
use label::Label;
use tch_goodies::Pixel;

/// The Pascal VOC dataset.
#[derive(Debug, Clone)]
pub struct VocDataset {
    pub classes: IndexSet<String>,
    pub samples: Vec<voc_dataset::Sample>,
    pub records: Vec<Arc<FileRecord>>,
}

impl GenericDataset for VocDataset {
    fn input_channels(&self) -> usize {
        3
    }

    fn classes(&self) -> &IndexSet<String> {
        &self.classes
    }
}

impl FileDataset for VocDataset {
    fn records(&self) -> &[Arc<FileRecord>] {
        &self.records
    }
}

impl VocDataset {
    pub async fn load(
        dataset_dir: impl AsRef<Path>,
        classes_file: impl AsRef<Path>,
        class_whitelist: Option<HashSet<String>>,
    ) -> Result<VocDataset> {
        let dataset_dir = dataset_dir.as_ref().to_owned();
        let classes_file = classes_file.as_ref();

        // load classes file
        let classes = load_classes_file(&classes_file).await?;

        // load samples
        let classes = Arc::new(classes);

        let (samples, records) = {
            let classes = classes.clone();

            tokio::task::spawn_blocking(move || -> Result<_> {
                let samples = voc_dataset::load(dataset_dir)?;

                // build records
                let records: Vec<_> = samples
                    .iter()
                    .map(|sample| -> Result<_> {
                        let voc_dataset::Sample {
                            image_path,
                            annotation,
                        } = sample;

                        let size = {
                            let voc_dataset::Size { width, height, .. } = annotation.size;
                            Pixel(HW::from_hw([height, width]))
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
                            .map(|(obj, class_index)| -> Result<_> {
                                let voc_dataset::BndBox {
                                    xmin,
                                    ymin,
                                    xmax,
                                    ymax,
                                } = obj.bndbox;
                                let bbox = Pixel(Label {
                                    rect: CyCxHW::from_tlbr([ymin, xmin, ymax, xmax]),
                                    class: class_index,
                                });
                                Ok(bbox)
                            })
                            .try_collect()?;

                        Ok(Arc::new(FileRecord {
                            path: image_path.clone(),
                            size,
                            bboxes,
                        }))
                    })
                    .try_collect()?;

                Ok((samples, records))
            })
            .await??
        };

        let classes = Arc::try_unwrap(classes).unwrap();

        Ok(VocDataset {
            classes,
            samples,
            records,
        })
    }
}
