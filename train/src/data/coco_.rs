use super::*;
use crate::{
    common::*,
    config::{Config, DatasetConfig},
};

#[derive(Debug, Clone)]
pub struct CocoDataset {
    pub config: Arc<Config>,
    pub dataset: coco::DataSet,
    pub category_id_to_classes: HashMap<usize, String>,
    pub classes: IndexSet<String>,
    pub records: Vec<Arc<FileRecord>>,
}

impl GenericDataset for CocoDataset {
    fn input_channels(&self) -> usize {
        3
    }

    // fn num_classes(&self) -> usize {
    //     self.classes.len()
    // }

    fn classes(&self) -> &IndexSet<String> {
        &self.classes
    }

    // fn num_records(&self) -> usize {
    //     self.records.len()
    // }
}

impl FileDataset for CocoDataset {
    fn records(&self) -> &[Arc<FileRecord>] {
        &self.records
    }
}

impl CocoDataset {
    pub async fn load(config: Arc<Config>, dir: &Path, name: &str) -> Result<CocoDataset> {
        let Config {
            dataset: DatasetConfig { classes_file, .. },
            ..
        } = &*config;
        let classes = load_classes_file(classes_file).await?;
        let dataset = coco::DataSet::load_async(dir, name).await?;
        let category_id_to_classes: HashMap<_, _> = dataset
            .instances
            .categories
            .iter()
            .map(|cat| {
                let coco::Category { id, ref name, .. } = *cat;
                (id, name.to_owned())
            })
            .collect();

        // sanity check
        {
            let categories: HashSet<_> = category_id_to_classes.values().collect();
            let classes: HashSet<_> = classes.iter().collect();
            let nonexist_classes: Vec<_> = classes.difference(&categories).collect();
            let uncovered_classes: Vec<_> = categories.difference(&classes).collect();

            if !nonexist_classes.is_empty() {
                warn!(
                    "these classes are not defined in dataset: {:?}",
                    nonexist_classes
                );
            }

            if !uncovered_classes.is_empty() {
                warn!(
                    "these classes are not covered by classes file: {:?}",
                    uncovered_classes
                );
            }
        }

        dataset.instances.annotations.iter().try_for_each(|ann| {
            ensure!(
                category_id_to_classes.contains_key(&ann.category_id),
                "invalid category id found"
            );
            Ok(())
        })?;

        // build records
        let annotations: HashMap<_, _> = dataset
            .instances
            .annotations
            .iter()
            .map(|ann| (ann.id, ann))
            .collect();
        let images: HashMap<_, _> = dataset
            .instances
            .images
            .iter()
            .map(|img| (img.id, img))
            .collect();
        let records = annotations
            .iter()
            .map(|(_id, ann)| (ann.image_id, ann))
            .into_group_map()
            .into_iter()
            .map(|(image_id, anns)| {
                let image = &images[&image_id];
                let bboxes = anns
                    .into_iter()
                    .filter_map(|ann| {
                        // filter by class list and whitelist
                        let category_name = &category_id_to_classes[&ann.category_id];
                        let class_index = classes.get_index_of(category_name)?;
                        if let Some(whitelist) = &config.dataset.class_whitelist {
                            whitelist.get(category_name)?;
                        }

                        let [l, t, w, h] = ann.bbox;
                        let bbox = PixelBBox::from_tlhw([t.into(), l.into(), h.into(), w.into()]);
                        Some(LabeledPixelBBox {
                            bbox,
                            category_id: class_index,
                        })
                    })
                    .collect_vec();

                Arc::new(FileRecord {
                    path: dataset.image_dir.join(&image.file_name),
                    size: PixelSize::new(image.height, image.width),
                    bboxes,
                })
            })
            .collect_vec();

        Ok(CocoDataset {
            config,
            dataset,
            category_id_to_classes,
            classes,
            records,
        })
    }
}
