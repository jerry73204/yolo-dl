use super::*;
use crate::common::*;

#[derive(Debug)]
pub struct CachedDataset {
    cache_loader: Arc<CacheLoader>,
    dataset: Arc<Box<dyn FileDataset>>,
}

impl CachedDataset {
    pub async fn new<P>(
        dataset: Arc<Box<dyn FileDataset>>,
        cache_dir: P,
        image_size: usize,
        device: Device,
    ) -> Result<Self>
    where
        P: AsRef<async_std::path::Path>,
    {
        let cache_loader = Arc::new(
            CacheLoader::new(cache_dir, image_size, dataset.input_channels(), device).await?,
        );
        Ok(Self {
            cache_loader,
            dataset,
        })
    }
}

impl GenericDataset for CachedDataset {
    fn input_channels(&self) -> usize {
        self.dataset.input_channels()
    }

    fn classes(&self) -> &IndexSet<String> {
        self.dataset.classes()
    }
}

impl RandomAccessDataset for CachedDataset {
    fn num_records(&self) -> usize {
        self.dataset.records().len()
    }

    fn nth(&self, index: usize) -> Box<dyn Future<Output = Result<DataRecord>>> {
        let record = self.dataset.records().get(index).cloned();
        let cache_loader = self.cache_loader.clone();

        Box::new(async move {
            let FileRecord { path, bboxes, .. } =
                &*record.ok_or_else(|| format_err!("invalid index {}", index))?;

            let (image, bboxes) = cache_loader
                .load_cache(path, bboxes)
                .await
                .with_context(|| format!("failed to load image file {}", path.display()))?;

            Ok(DataRecord { image, bboxes })
        })
    }
}
