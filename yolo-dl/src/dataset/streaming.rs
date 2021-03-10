use super::*;
use crate::common::*;

#[derive(Debug)]
pub struct RandomAccessStream<D>
where
    D: 'static + RandomAccessDataset + Sync,
{
    dataset: Arc<D>,
}

impl<D> RandomAccessStream<D>
where
    D: RandomAccessDataset + Sync,
{
    pub fn new(dataset: D) -> Self {
        Self {
            dataset: Arc::new(dataset),
        }
    }
}

impl<D> GenericDataset for RandomAccessStream<D>
where
    D: 'static + RandomAccessDataset + Sync,
{
    fn input_channels(&self) -> usize {
        self.dataset.input_channels()
    }

    fn classes(&self) -> &IndexSet<String> {
        self.dataset.classes()
    }
}

impl<D> StreamingDataset for RandomAccessStream<D>
where
    D: 'static + RandomAccessDataset + Sync,
{
    fn stream(&self) -> Result<Pin<Box<dyn Stream<Item = Result<DataRecord>> + Send>>> {
        let num_records = self.dataset.num_records();
        let dataset = self.dataset.clone();
        let stream = stream::iter(0..num_records).then(move |index| {
            let dataset = dataset.clone();
            async move {
                let record = dataset.nth(index).await?;
                Ok(record)
            }
        });

        Ok(Box::pin(stream))
    }
}
