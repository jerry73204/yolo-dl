use super::*;
use crate::common::*;

/// The generic dataset trait.
pub trait GenericDataset
where
    Self: Debug + Send,
{
    /// The number of color channels of the dataset.
    fn input_channels(&self) -> usize;

    /// The list of class names of the dataset.
    fn classes(&self) -> &IndexSet<String>;
}

/// The dataset with a list of image paths.
pub trait FileDataset
where
    Self: GenericDataset,
{
    /// Get the list of image paths in the dataset.
    fn records(&self) -> &[Arc<FileRecord>];
}

/// The dataset that can be random accessed.
pub trait RandomAccessDataset
where
    Self: GenericDataset,
{
    /// Get number of records in the dataset.
    fn num_records(&self) -> usize;

    /// Get the nth record in the dataset.
    fn nth(&self, index: usize) -> Pin<Box<dyn Future<Output = Result<DataRecord>> + Send>>;
}

/// The dataset that can be enumerated through a stream.
pub trait StreamingDataset
where
    Self: GenericDataset,
{
    fn stream(&self) -> Result<Pin<Box<dyn Stream<Item = Result<DataRecord>> + Send>>>;
}
