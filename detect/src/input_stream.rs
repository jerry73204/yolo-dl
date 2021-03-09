// use crate::common::*;

// #[derive(Debug)]
// pub struct InputStream {
//     config: Arc<Config>,
//     dataset: Arc<Box<dyn RandomAccessDataset>>,
// }

// impl InputStream {
//     pub async fn new(config: Arc<Config>) -> Result<Self> {
//         let dataset = {
//             let Config {
//                 dataset:
//                     DatasetConfig {
//                         ref kind,
//                         ref class_whitelist,
//                         ..
//                     },
//                 ..
//             } = *config;

//             match *kind {
//                 DatasetKind::Coco {
//                     ref dataset_dir,
//                     ref classes_file,
//                     ref dataset_name,
//                     image_size,
//                     ..
//                 } => {
//                     let dataset = CocoDataset::load(
//                         dataset_dir,
//                         classes_file,
//                         class_whitelist.clone(),
//                         dataset_name,
//                     )
//                     .await?;
//                     let dataset =
//                         SanitizedDataset::new(dataset, out_of_bound_tolerance, min_bbox_size)?;
//                     let dataset =
//                         CachedDataset::new(dataset, cache_dir, image_size.get(), device).await?;
//                     let dataset: Box<dyn RandomAccessDataset> = Box::new(dataset);
//                     dataset
//                 }
//                 DatasetKind::Voc {
//                     ref dataset_dir,
//                     ref classes_file,
//                     image_size,
//                     ..
//                 } => {
//                     let dataset =
//                         VocDataset::load(dataset_dir, classes_file, class_whitelist.clone())
//                             .await?;
//                     let dataset =
//                         SanitizedDataset::new(dataset, out_of_bound_tolerance, min_bbox_size)?;
//                     let dataset =
//                         CachedDataset::new(dataset, cache_dir, image_size.get(), device).await?;
//                     let dataset: Box<dyn RandomAccessDataset> = Box::new(dataset);
//                     dataset
//                 }
//                 DatasetKind::Iii {
//                     ref dataset_dir,
//                     ref classes_file,
//                     ref blacklist_files,
//                     image_size,
//                     ..
//                 } => {
//                     let dataset = IiiDataset::load(
//                         dataset_dir,
//                         classes_file,
//                         class_whitelist.clone(),
//                         blacklist_files.clone(),
//                     )
//                     .await?;
//                     let dataset =
//                         SanitizedDataset::new(dataset, out_of_bound_tolerance, min_bbox_size)?;
//                     let dataset =
//                         CachedDataset::new(dataset, cache_dir, image_size.get(), device).await?;
//                     let dataset: Box<dyn RandomAccessDataset> = Box::new(dataset);
//                     dataset
//                 }
//                 DatasetKind::Csv {
//                     ref image_dir,
//                     ref label_file,
//                     ref classes_file,
//                     image_size,
//                     input_channels,
//                     ..
//                 } => {
//                     let dataset = CsvDataset::load(
//                         image_dir,
//                         label_file,
//                         classes_file,
//                         input_channels.get(),
//                         class_whitelist.clone(),
//                     )
//                     .await?;
//                     let dataset =
//                         SanitizedDataset::new(dataset, out_of_bound_tolerance, min_bbox_size)?;
//                     let dataset =
//                         CachedDataset::new(dataset, cache_dir, image_size.get(), device).await?;
//                     let dataset: Box<dyn RandomAccessDataset> = Box::new(dataset);
//                     dataset
//                 }
//             }
//         };

//         Ok(Self {})
//     }
// }
