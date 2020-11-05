use super::*;
use crate::{
    common::*,
    config::{Config, DatasetConfig},
};

#[derive(Debug, Clone)]
pub struct VocDataset {
    pub config: Arc<Config>,
    pub classes: IndexSet<String>,
    pub samples: Vec<voc_dataset::Sample>,
}

impl GenericDataset for VocDataset {
    fn input_channels(&self) -> usize {
        3
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
                let voc_dataset::Sample {
                    image_path,
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
                    path: image_path.clone(),
                    size,
                    bboxes,
                }))
            })
            .try_collect()?;

        Ok(records)
    }
}

impl VocDataset {
    pub async fn load<P>(config: Arc<Config>, dataset_dir: P) -> Result<VocDataset>
    where
        P: AsRef<Path>,
    {
        let Config {
            dataset: DatasetConfig { classes_file, .. },
            ..
        } = &*config;
        let dataset_dir = dataset_dir.as_ref().to_owned();

        // load classes file
        let classes = load_classes_file(&classes_file).await?;

        // load samples
        let samples =
            async_std::task::spawn_blocking(move || voc_dataset::load(dataset_dir)).await?;

        Ok(VocDataset {
            config,
            classes,
            samples,
        })
    }
}
