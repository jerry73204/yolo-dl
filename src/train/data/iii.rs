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
                    let annotation: IiiAnnotation = serde_xml_rs::from_str(&xml_content)
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
    pub annotation: IiiAnnotation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "annotation")]
pub struct IiiAnnotation {
    pub filename: String,
    pub folder: String,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub object: Vec<voc_dataset::Object>,
    pub segmented: Option<bool>,
    pub size: voc_dataset::Size,
}

// /// Correspond to <pose> in annotation XML.
// #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
// #[serde(rename = "pose")]
// pub enum Pose {
//     Frontal,
//     Rear,
//     Left,
//     Right,
//     Unspecified,
// }

// /// Correspond to <bndbox> in annotation XML.
// #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
// #[serde(rename = "bndbox")]
// pub struct BndBox {
//     pub xmin: R64,
//     pub ymin: R64,
//     pub xmax: R64,
//     pub ymax: R64,
// }

// /// Correspond to <size> in annotation XML.
// #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
// #[serde(rename = "size")]
// pub struct Size {
//     pub width: usize,
//     pub height: usize,
//     pub depth: usize,
// }

// /// Correspond to <size> in annotation XML.
// #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
// #[serde(rename = "point")]
// pub struct Point {
//     pub x: usize,
//     pub y: usize,
// }

// /// Correspond to <object> in annotation XML.
// #[derive(Debug, Clone, Serialize, Deserialize)]
// #[serde(rename = "object")]
// pub struct Object {
//     pub name: String,
//     pub pose: Pose,
//     pub bndbox: BndBox,
//     pub actions: Option<Actions>,
//     #[serde(skip_serializing_if = "Vec::is_empty", default)]
//     pub part: Vec<Part>,
//     pub truncated: Option<bool>,
//     pub difficult: Option<bool>,
//     pub occluded: Option<bool>,
//     pub point: Option<Point>,
// }

// /// Correspond to <part> in annotation XML.
// #[derive(Debug, Clone, Serialize, Deserialize)]
// #[serde(rename = "part")]
// pub struct Part {
//     pub name: String,
//     pub bndbox: BndBox,
// }

// /// Correspond to <source> in annotation XML.
// #[derive(Debug, Clone, Serialize, Deserialize)]
// #[serde(rename = "source")]
// pub struct Source {
//     pub database: String,
//     pub annotation: String,
//     pub image: String,
// }

// #[derive(Debug, Clone, Serialize, Deserialize)]
// #[serde(rename = "actions")]
// pub struct Actions {
//     pub jumping: bool,
//     pub other: bool,
//     pub phoning: bool,
//     pub playinginstrument: bool,
//     pub reading: bool,
//     pub ridinghorse: bool,
//     pub running: bool,
//     pub takingphoto: bool,
//     pub walking: bool,
// }
