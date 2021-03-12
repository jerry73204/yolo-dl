use super::*;
use crate::{common::*, processor::MemoryCache};

/// The dataset backed with image data caching.
#[derive(Debug)]
pub struct MemoryCacheDataset<D>
where
    D: FileDataset,
{
    cache_loader: Arc<MemoryCache>,
    dataset: D,
}

impl<D> MemoryCacheDataset<D>
where
    D: FileDataset,
{
    pub async fn new(dataset: D, image_size: usize, device: Device) -> Result<Self> {
        let cache_loader =
            Arc::new(MemoryCache::new(image_size, dataset.input_channels(), device).await?);
        Ok(Self {
            cache_loader,
            dataset,
        })
    }
}

impl<D> GenericDataset for MemoryCacheDataset<D>
where
    D: FileDataset,
{
    fn input_channels(&self) -> usize {
        self.dataset.input_channels()
    }

    fn classes(&self) -> &IndexSet<String> {
        self.dataset.classes()
    }
}

impl<D> RandomAccessDataset for MemoryCacheDataset<D>
where
    D: FileDataset,
{
    fn num_records(&self) -> usize {
        self.dataset.records().len()
    }

    fn nth(&self, index: usize) -> Pin<Box<dyn Future<Output = Result<DataRecord>> + Send>> {
        let record = self.dataset.records().get(index).cloned();
        let cache_loader = self.cache_loader.clone();

        Box::pin(async move {
            let FileRecord {
                path, bboxes, size, ..
            } = &*record.ok_or_else(|| format_err!("invalid index {}", index))?;

            let (image, bboxes) = cache_loader
                .load_cache(path, size, bboxes)
                .await
                .with_context(|| format!("failed to load image file {}", path.display()))?;

            Ok(DataRecord { image, bboxes })
        })
    }
}
