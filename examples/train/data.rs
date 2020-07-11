use crate::{common::*, config::Config};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Record {
    pub path: PathBuf,
    pub bbox: [R64; 4],
    pub category_id: usize,
    pub height: usize,
    pub width: usize,
}

pub async fn train_stream(config: Arc<Config>) -> Result<(Vec<Record>, HashMap<usize, Category>)> {
    let (records, categories) = {
        let Config {
            dataset_dir,
            dataset_name,
        } = &*config;

        let DataSet {
            instances,
            image_dir,
            ..
        } = DataSet::load_async(dataset_dir, &dataset_name).await?;

        let annotations = instances
            .annotations
            .into_iter()
            .map(|ann| (ann.id, ann))
            .collect::<HashMap<_, _>>();
        let images = instances
            .images
            .into_iter()
            .map(|img| (img.id, img))
            .collect::<HashMap<_, _>>();
        let categories = instances
            .categories
            .into_iter()
            .map(|cat| (cat.id, cat))
            .collect::<HashMap<_, _>>();

        let records = annotations
            .iter()
            .map(|(id, ann)| {
                let image = &images[&ann.image_id];

                Record {
                    path: image_dir.join(&image.file_name),
                    height: image.height,
                    width: image.width,
                    bbox: ann.bbox.clone(),
                    category_id: ann.category_id,
                }
            })
            .collect::<Vec<_>>();

        (records, categories)
    };

    Ok((records, categories))
}
