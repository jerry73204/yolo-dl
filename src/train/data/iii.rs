use super::*;
use crate::common::*;

pub async fn load_iii_dataset<P>(dataset_dir: P) -> Result<Vec<IiiSample>>
where
    P: AsRef<Path>,
{
    let dataset_dir = dataset_dir.as_ref();

    let xml_files = {
        let dataset_dir = dataset_dir.to_owned();
        async_std::task::spawn_blocking(move || {
            let xml_files: Vec<_> =
                glob::glob(&format!("{}/**/*.xml", dataset_dir.display()))?.try_collect()?;
            Fallible::Ok(xml_files)
        })
        .await?
    };

    let tasks = xml_files
        .into_iter()
        .map(|xml_file| async move {
            let xml_file = Arc::new(xml_file);

            let xml_content = async_std::fs::read_to_string(&*xml_file)
                .await
                .with_context(|| {
                    format!("failed to read annotation file {}", xml_file.display())
                })?;

            let annotation = {
                let xml_file = xml_file.clone();
                async_std::task::spawn_blocking(move || -> Result<_> {
                    let annotation: voc_dataset::Annotation = serde_xml_rs::from_str(&xml_content)
                        .with_context(|| {
                            format!("failed to parse annotation file {}", xml_file.display())
                        })?;
                    Ok(annotation)
                })
                .await?
            };

            let image_file = {
                let file_name = format!("{}.jpg", xml_file.file_stem().unwrap().to_str().unwrap());
                xml_file
                    .parent()
                    .ok_or_else(|| format_err!("invalid xml path {}", xml_file.display()))?
                    .join(file_name)
            };

            let sample = IiiSample {
                annotation,
                image_file,
            };

            Fallible::Ok(sample)
        })
        .map(async_std::task::spawn);

    let samples = future::try_join_all(tasks).await?;
    Ok(samples)
}

#[derive(Debug, Clone)]
pub struct IiiSample {
    pub image_file: PathBuf,
    pub annotation: voc_dataset::Annotation,
}
