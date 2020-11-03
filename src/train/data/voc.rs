use super::*;
use crate::{common::*, config::Config};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VocDataset {
    pub samples: Vec<voc_dataset::Sample>,
}

impl GenericDataset for VocDataset {
    fn input_channels(&self) -> usize {
        3
    }

    fn num_classes(&self) -> usize {
        todo!();
    }

    fn records(&self) -> Result<Vec<Arc<DataRecord>>> {
        todo!()
    }
}

impl VocDataset {
    pub async fn load<P>(config: Arc<Config>, dataset_dir: P) -> Result<VocDataset>
    where
        P: AsRef<Path>,
    {
        let dataset_dir = dataset_dir.as_ref().to_owned();
        let samples =
            async_std::task::spawn_blocking(move || voc_dataset::load(dataset_dir)).await?;
        Ok(VocDataset { samples })
    }
}
