use crate::common::*;

pub async fn load_voc_dataset<P>(dataset_dir: P) -> Result<Vec<voc_dataset::Sample>>
where
    P: AsRef<Path>,
{
    let dataset_dir = dataset_dir.as_ref().to_owned();
    let samples = async_std::task::spawn_blocking(move || voc_dataset::load(dataset_dir)).await?;
    Ok(samples)
}
