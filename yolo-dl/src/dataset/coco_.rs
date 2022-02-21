use super::*;
use crate::common::*;
use bbox::{prelude::*, CyCxHW, HW};
use label::Label;
use tch_goodies::Pixel;

/// The Microsoft COCO dataset.
#[derive(Debug, Clone)]
pub struct CocoDataset {
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
    pub async fn load(
        dataset_dir: impl AsRef<Path>,
        classes_file: impl AsRef<Path>,
        class_whitelist: Option<HashSet<String>>,
        name: &str,
    ) -> Result<CocoDataset> {
        let dataset_dir = dataset_dir.as_ref();
        let classes_file = classes_file.as_ref();

        let classes = load_classes_file(classes_file).await?;
        let dataset = coco::DataSet::load_async(dataset_dir, name).await?;
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
        let records: Vec<_> = annotations
            .iter()
            .map(|(_id, ann)| (ann.image_id, ann))
            .into_group_map()
            .into_iter()
            .map(|(image_id, anns)| -> Result<_> {
                let image = &images[&image_id];
                let bboxes: Vec<_> = anns
                    .into_iter()
                    .map(|ann| -> Result<_> {
                        // filter by class list and whitelist
                        let category_name = &category_id_to_classes[&ann.category_id];
                        let class_index = match classes.get_index_of(category_name) {
                            Some(index) => index,
                            None => return Ok(None),
                        };
                        if let Some(whitelist) = &class_whitelist {
                            if whitelist.get(category_name).is_none() {
                                return Ok(None);
                            }
                        }

                        let [l, t, w, h] = ann.bbox;
                        let rect = CyCxHW::from_tlhw([t, l, h, w]);
                        Ok(Some(Pixel(Label {
                            rect,
                            class: class_index,
                        })))
                    })
                    .filter_map(|result| result.transpose())
                    .try_collect()?;

                Ok(Arc::new(FileRecord {
                    path: dataset.image_dir.join(&image.file_name),
                    size: Pixel(HW::from_hw([image.height, image.width])),
                    bboxes,
                }))
            })
            .try_collect()?;

        Ok(CocoDataset {
            dataset,
            category_id_to_classes,
            classes,
            records,
        })
    }
}
