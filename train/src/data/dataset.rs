use super::*;
use crate::common::*;

pub trait GenericDataset
where
    Self: Debug + Sync + Send,
{
    fn input_channels(&self) -> usize;
    fn classes(&self) -> &IndexSet<String>;
}

pub trait FileDataset
where
    Self: GenericDataset,
{
    fn records(&self) -> &[Arc<FileRecord>];
}

pub trait RandomAccessDataset
where
    Self: GenericDataset,
{
    fn num_records(&self) -> usize;
    fn nth(&self, index: usize) -> Pin<Box<dyn Future<Output = Result<DataRecord>> + Send>>;
}
