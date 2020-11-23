pub mod cache;
mod common;
pub mod utils;

use crate::{
    cache::{BBoxEntry, ClassEntry, DatasetWriterInit, Header, ImageEntry, ImageItem},
    common::*,
};

#[derive(Debug, Clone, FromArgs)]
/// Dataset cache toolkit
struct Args {
    #[argh(subcommand)]
    pub subcommand: SubCommand,
}

#[derive(Debug, Clone, FromArgs)]
#[argh(subcommand)]
enum SubCommand {
    Info(InfoArgs),
    Build(BuildArgs),
}

/// Query cache file information
#[derive(Debug, Clone, FromArgs)]
#[argh(subcommand, name = "info")]
struct InfoArgs {
    #[argh(positional)]
    pub cache_file: PathBuf,
}

/// Build cache for dataset
#[derive(Debug, Clone, FromArgs)]
#[argh(subcommand, name = "build")]
struct BuildArgs {
    #[argh(positional)]
    pub kind: String,
    #[argh(positional)]
    pub image_size: String,
    #[argh(positional)]
    pub classes_file: PathBuf,
    #[argh(positional)]
    pub dataset_dir: PathBuf,
    #[argh(positional)]
    pub output_file: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IiiSample {
    pub image_file: PathBuf,
    pub annotation_file: PathBuf,
    pub annotation: voc::Annotation,
}

#[async_std::main]
async fn main() -> Result<()> {
    let args: Args = argh::from_env();

    match args.subcommand {
        SubCommand::Info(info_args) => {
            info(info_args).await?;
        }
        SubCommand::Build(build_args) => {
            build(build_args).await?;
        }
    }

    Ok(())
}

async fn info(args: InfoArgs) -> Result<()> {
    let InfoArgs { cache_file } = args;

    // create memory mapped file
    let mmap = unsafe {
        let output_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&cache_file)?;
        let mmap = MmapOptions::new().map_mut(&output_file)?;
        mmap
    };

    async_std::task::spawn_blocking(move || -> Result<_> {
        let mmap_slice = mmap.as_ref();
        let mut cursor = std::io::Cursor::new(mmap_slice);

        // deserialize header
        let Header {
            magic,
            num_classes,
            num_images,
            alignment,
            shape,
            component_kind,
            data_offset,
            bbox_offset,
        } = bincode::deserialize_from(&mut cursor)?;
        let component_size = component_kind.component_size();

        ensure!(magic == cache::MAGIC, "file magic does not match");

        // deserialize class entries
        let classes: IndexMap<_, _> = (0..num_classes)
            .map(|_| -> Result<_> {
                let ClassEntry { index, name } = bincode::deserialize_from(&mut cursor)?;
                Ok((index, name))
            })
            .try_collect()?;

        // deserialize image entries
        let image_entries: Vec<_> = (0..num_images)
            .map(|_| -> Result<_> {
                let entry: ImageEntry = bincode::deserialize_from(&mut cursor)?;
                Ok(entry)
            })
            .try_collect()?;

        // calculate data and bbox section offsets
        let per_image_size = {
            let [c, h, w] = shape;
            component_size * c as usize * h as usize * w as usize
        };
        let per_data_size = utils::nearest_multiple(per_image_size, alignment as usize);

        let bbox_ranges = image_entries.iter().scan(0usize, |bbox_index, entry| {
            let ImageEntry { num_bboxes, .. } = *entry;
            let begin = *bbox_index;
            let end = begin + num_bboxes as usize;
            *bbox_index = end;
            Some(begin..end)
        });

        // deserialize bboxes
        let num_bboxes = bbox_ranges.last().map(|range| range.end).unwrap_or(0);

        cursor.set_position(bbox_offset as u64);
        let bbox_entries: Vec<_> = (0..num_bboxes)
            .map(|_| -> Result<_> {
                let bbox_entry: BBoxEntry = bincode::deserialize_from(&mut cursor)?;
                Ok(bbox_entry)
            })
            .try_collect()?;

        Ok(())
    })
    .await?;

    Ok(())
}

async fn build(args: BuildArgs) -> Result<()> {
    let BuildArgs {
        kind,
        image_size,
        classes_file,
        dataset_dir,
        output_file,
    } = args;

    let image_size = {
        let tokens: Vec<_> = image_size.split('x').collect();
        let sizes: Vec<usize> = tokens
            .into_iter()
            .map(|token| token.parse())
            .try_collect()
            .with_context(|| "the image size must be integers")?;
        let sizes = <[usize; 3]>::try_from(sizes)
            .map_err(|_| format_err!("the image size must be CxHxW"))?;
        sizes
    };

    match kind.as_str() {
        "coco" => {}
        "iii" => {
            build_iii_dataset(
                image_size,
                &classes_file,
                &dataset_dir,
                output_file,
                HashSet::new(),
            )
            .await?;
        }
        "voc" => {}
        _ => bail!("dataset kind '{}' is not supported"),
    }

    Ok(())
}

async fn build_iii_dataset(
    image_size: [usize; 3],
    classes_file: impl AsRef<async_std::path::Path>,
    dataset_dir: impl AsRef<async_std::path::Path>,
    output_file: impl AsRef<async_std::path::Path>,
    blacklist_files: HashSet<PathBuf>,
) -> Result<()> {
    let classes = {
        let classes = load_classes_file(&classes_file).await?;
        Arc::new(classes)
    };
    let [target_c, target_h, target_w] = image_size;
    let bar = Arc::new(ProgressBar::new(0).with_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
            )
            .progress_chars("#>-"),
    ));

    // list xml files
    let xml_files = {
        bar.println("listing annotation files...");
        let bar = bar.clone();
        let dataset_dir = dataset_dir.as_ref().to_owned();

        async_std::task::spawn_blocking(move || {
            let xml_files: Vec<_> = glob::glob(&format!("{}/**/*.xml", dataset_dir.display()))?
                .map(|result| -> Result<_> {
                    let path = result?;
                    let suffix = path.strip_prefix(&dataset_dir).unwrap();
                    if blacklist_files.contains(suffix) {
                        warn!("ignore blacklisted file '{}'", path.display());
                        Ok(None)
                    } else {
                        Ok(Some(path))
                    }
                })
                .filter_map(|result| result.transpose())
                .inspect(|_| {
                    bar.inc_length(1);
                })
                .try_collect()?;
            Fallible::Ok(xml_files)
        })
        .await?
    };

    // parse xml files
    let mut samples: Vec<_> = {
        bar.println(format!("{} annotation files found", xml_files.len()));
        bar.println("parsing annotation files...");
        let bar_clone = bar.clone();

        let samples: Vec<_> = stream::iter(xml_files.into_iter())
            .par_then(None, move |annotation_file| {
                let bar = bar_clone.clone();

                async move {
                    let xml_content = async_std::fs::read_to_string(&*annotation_file)
                        .await
                        .with_context(|| {
                            format!(
                                "failed to read annotation file {}",
                                annotation_file.display()
                            )
                        })?;

                    let annotation: voc::Annotation = {
                        async_std::task::spawn_blocking(move || {
                            serde_xml_rs::from_str(&xml_content)
                        })
                        .await
                        .with_context(|| {
                            format!(
                                "failed to parse annotation file {}",
                                annotation_file.display()
                            )
                        })?
                    };

                    let image_file = {
                        let file_name = format!(
                            "{}.jpg",
                            annotation_file.file_stem().unwrap().to_str().unwrap()
                        );
                        let image_file = annotation_file.parent().unwrap().join(file_name);
                        image_file
                    };

                    // sanity check
                    ensure!(
                        annotation.size.depth == 3,
                        "expect depth to be {}, but found {}",
                        3,
                        annotation.size.depth
                    );

                    let sample = IiiSample {
                        annotation,
                        annotation_file,
                        image_file,
                    };

                    bar.inc(1);

                    Fallible::Ok(sample)
                }
            })
            .try_collect()
            .await?;

        bar.finish();

        samples
    };

    samples.sort_by_cached_key(|sample| sample.annotation_file.clone());
    let num_images = samples.len();

    // build cache meta
    bar.reset();
    bar.set_length(samples.len() as u64);
    bar.println("process images...");

    let image_stream = {
        let bar_clone = bar.clone();
        let classes = classes.clone();

        stream::iter(samples.into_iter()).par_then(None, move |sample| {
            let classes = classes.clone();
            let bar = bar_clone.clone();

            async move {
                let IiiSample {
                    annotation:
                        voc::Annotation {
                            object,
                            size: voc::Size { depth: orig_c, .. },
                            ..
                        },
                    image_file,
                    ..
                } = sample;
                ensure!(
                    orig_c == target_c,
                    "expcet target number of channels {}, but found {}",
                    target_c,
                    orig_c
                );

                // build bbox entries
                let bboxes = object.into_iter().filter_map(move |obj| {
                    let voc::Object {
                        name,
                        bndbox:
                            voc::BndBox {
                                ymin,
                                ymax,
                                xmin,
                                xmax,
                            },
                        ..
                    } = obj;
                    let tlbr = [ymin.raw(), xmin.raw(), ymax.raw(), xmax.raw()];
                    let class_index = classes.get_index_of(&name)?;

                    Some(BBoxEntry {
                        tlbr,
                        class: class_index as u32,
                    })
                });

                let data = async_std::task::spawn_blocking(move || -> Result<_> {
                    let image = vision::image::load(&image_file)?
                        .resize2d_letterbox(target_h as i64, target_w as i64)?
                        .to_kind(Kind::Float)
                        .g_div1(255.0);
                    assert_eq!(
                        image.size3()?,
                        (target_c as i64, target_h as i64, target_w as i64)
                    );
                    let data = unsafe {
                        let numel = image.numel();
                        let mut data: Vec<f32> = Vec::with_capacity(numel);
                        let slice = slice::from_raw_parts_mut(data.as_mut_ptr(), numel);
                        image.copy_data(slice, numel);
                        data.set_len(numel);
                        data
                    };

                    Ok(data)
                })
                .await?;

                bar.inc(1);

                Fallible::Ok(ImageItem { data, bboxes })
            }
        })
    };

    {
        let classes: IndexMap<_, _> = classes
            .iter()
            .enumerate()
            .map(|(index, class)| (index as u32, class.clone()))
            .collect();
        let shape = [target_c as u32, target_h as u32, target_w as u32];

        DatasetWriterInit {
            num_images,
            shape,
            alignment: None,
            classes,
            images: image_stream,
        }
        .write(&output_file)
        .await?;
    }

    bar.finish();

    Ok(())
}

async fn load_classes_file<P>(path: P) -> Result<IndexSet<String>>
where
    P: AsRef<async_std::path::Path>,
{
    let path = path.as_ref();
    let content = async_std::fs::read_to_string(path).await?;
    let lines: Vec<_> = content.lines().collect();
    let classes: IndexSet<_> = lines.iter().cloned().map(ToOwned::to_owned).collect();
    ensure!(
        lines.len() == classes.len(),
        "duplicated class names found in '{}'",
        path.display()
    );
    ensure!(
        classes.len() > 0,
        "no classes found in '{}'",
        path.display()
    );
    Ok(classes)
}
