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
}

impl GenericDataset for CocoDataset {
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
        let CocoDataset {
            classes,
            category_id_to_classes,
            dataset:
                coco::DataSet {
                    instances,
                    image_dir,
                    ..
                },
            ..
        } = self;

        let annotations: HashMap<_, _> = instances
            .annotations
            .iter()
            .map(|ann| (ann.id, ann))
            .collect();
        let images: HashMap<_, _> = instances.images.iter().map(|img| (img.id, img)).collect();
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
                        if let Some(whitelist) = &self.config.dataset.class_whiltelist {
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

                Arc::new(DataRecord {
                    path: image_dir.join(&image.file_name),
                    size: PixelSize::new(image.height, image.width),
                    bboxes,
                })
            })
            .collect_vec();

        Ok(records)
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
        dataset.instances.annotations.iter().try_for_each(|ann| {
            ensure!(
                category_id_to_classes.contains_key(&ann.category_id),
                "invalid category id found"
            );
            Ok(())
        })?;

        Ok(CocoDataset {
            config,
            dataset,
            category_id_to_classes,
            classes,
        })
    }
}
