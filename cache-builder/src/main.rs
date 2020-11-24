pub mod cache;
mod common;
pub mod utils;

use crate::{
    cache::{BBoxEntry, ComponentKind, Dataset, DatasetWriterInit, Header, ImageItem},
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
    ExtractImage(ExtractImageArgs),
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

/// Extract images from cache file
#[derive(Debug, Clone, FromArgs)]
#[argh(subcommand, name = "extract_image")]
struct ExtractImageArgs {
    /// draw bounding boxes on images
    #[argh(switch)]
    pub draw_bbox: bool,
    #[argh(positional)]
    pub range: String,
    #[argh(positional)]
    pub cache_file: PathBuf,
    #[argh(positional)]
    pub output_dir: PathBuf,
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
        SubCommand::ExtractImage(extract_args) => {
            extract_images(extract_args).await?;
        }
    }

    Ok(())
}

async fn info(args: InfoArgs) -> Result<()> {
    let InfoArgs { cache_file } = args;

    // load dataset
    let Dataset {
        header:
            Header {
                magic: _,
                shape,
                component_kind,
                alignment,
                data_offset,
                bbox_offset,
            },
        classes,
        image_entries,
        bbox_entries,
        ..
    } = Dataset::open(cache_file).await?;

    // print classes
    println!("# header");

    let mut table = Table::new();
    table.add_row(row!["num_classes", classes.len()]);
    table.add_row(row!["num_images", image_entries.len()]);
    table.add_row(row!["num_bboxes", bbox_entries.len()]);
    table.add_row(row!["shape", format!("{:?}", shape)]);
    table.add_row(row!["component kind", format!("{:?}", component_kind)]);
    table.add_row(row!["alignment", alignment]);
    table.add_row(row!["data_offset", data_offset]);
    table.add_row(row!["bbox_offset", bbox_offset]);
    table.printstd();

    println!();
    println!("# classes");
    let table = classes
        .iter()
        .fold(Table::new(), |mut table, (index, name)| {
            table.add_row(row![index, name]);
            table
        });
    table.printstd();

    Ok(())
}

async fn extract_images(args: ExtractImageArgs) -> Result<()> {
    let ExtractImageArgs {
        draw_bbox,
        range,
        cache_file,
        output_dir,
    } = args;
    let output_dir = Arc::new(output_dir);

    // create output dir
    async_std::fs::create_dir_all(&*output_dir).await?;

    // load dataset
    let dataset = Arc::new(Dataset::open(cache_file).await?);
    let Dataset {
        header: Header {
            shape,
            component_kind,
            ..
        },
        ..
    } = *dataset;
    let shape = {
        let [c, h, w] = shape;
        [c as i64, h as i64, w as i64]
    };
    let num_images = dataset.image_iter().len();

    // update range
    // parse range command line argument
    let (range_begin, range_end) = {
        let range_error = "the range format must be one of .., BEGIN.., ..END, BEGIN..END";
        let mut tokens = range.split("..");
        let range_begin: usize = match tokens.next() {
            None => {
                bail!("{}", range_error)
            }
            Some("") => 0,
            Some(token) => {
                let value: usize = token.parse()?;
                ensure!(
                    value < num_images,
                    "the start index {} must not reach the number of images {}",
                    value,
                    num_images
                );
                value
            }
        };
        let range_end: usize = match tokens.next() {
            None => bail!("{}", range_error),
            Some("") => num_images,
            Some(token) => {
                let value: usize = token.parse()?;
                ensure!(
                    value <= num_images,
                    "the index {} must not exceed the number of images {}",
                    value,
                    num_images
                );
                value
            }
        };
        ensure!(matches!(tokens.next(), None), "{}", range_error);
        ensure!(range_begin <= range_end, "{} is not a valid range", range);

        (range_begin, range_end)
    };

    // save images
    let bar = Arc::new(ProgressBar::new((range_end - range_begin) as u64).with_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
            )
            .progress_chars("#>-"),
    ));
    let bar_clone = bar.clone();

    stream::iter(range_begin..range_end)
        .par_map(None, move |index| {
            let output_dir = output_dir.clone();
            let dataset = dataset.clone();
            let bar = bar_clone.clone();

            move || {
                let (data, bboxes) = dataset.image_iter().nth(index).unwrap();

                let tensor = match component_kind {
                    ComponentKind::F32 => {
                        let data: &[f32] = safe_transmute::transmute_many_pedantic(data).unwrap();
                        Tensor::of_slice(data)
                            .view(shape)
                            .g_mul1(255.0)
                            .to_kind(Kind::Uint8)
                    }
                    ComponentKind::F64 => {
                        let data: &[f64] = safe_transmute::transmute_many_pedantic(data).unwrap();
                        let tensor = Tensor::of_slice(data);
                        tensor.view(shape).g_mul1(255.0).to_kind(Kind::Uint8)
                    }
                    ComponentKind::U8 => {
                        let tensor = Tensor::of_slice(data).view(shape);
                        tensor.view(shape)
                    }
                };

                let tensor = if draw_bbox {
                    let [_channels, height, width] = shape;
                    let color = Tensor::of_slice(&[255u8, 255, 0]);

                    let tensor = bboxes.iter().fold(tensor, |mut tensor, bbox| {
                        let BBoxEntry {
                            tlbr: [bbox_t, bbox_l, bbox_b, bbox_r],
                            ..
                        } = *bbox;
                        let bbox_t = (bbox_t * height as f64) as i64;
                        let bbox_b = (bbox_b * height as f64) as i64;
                        let bbox_l = (bbox_l * width as f64) as i64;
                        let bbox_r = (bbox_r * width as f64) as i64;
                        tensor.draw_rect_(bbox_t, bbox_l, bbox_b, bbox_r, 1, &color)
                    });

                    tensor
                } else {
                    tensor
                };

                let output_path = output_dir.join(format!("{}.jpg", index));
                vision::image::save(&tensor, output_path)?;

                bar.inc(1);

                Fallible::Ok(())
            }
        })
        .try_for_each(|_: ()| async move { Fallible::Ok(()) })
        .await?;

    bar.finish();

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
        bar.println("scanning annotation files...");
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
                            size:
                                voc::Size {
                                    depth: orig_c,
                                    height: orig_h,
                                    width: orig_w,
                                },
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

                let scale = (target_h as f64 / orig_h as f64).min(target_w as f64 / orig_w as f64);
                let (margin_top, margin_left) = {
                    let (inner_h, inner_w) = (orig_h as f64 * scale, orig_w as f64 * scale);
                    let margin_top = (target_h as f64 - inner_h) / 2.0;
                    let margin_left = (target_w as f64 - inner_w) / 2.0;
                    (margin_top, margin_left)
                };

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

                    let tlbr = [
                        (ymin.raw() * scale + margin_top) / target_h as f64,
                        (xmin.raw() * scale + margin_left) / target_w as f64,
                        (ymax.raw() * scale + margin_top) / target_h as f64,
                        (xmax.raw() * scale + margin_left) / target_w as f64,
                    ];
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_test() -> Result<()> {
        let image_file = "/home/aeon/dataset/ntu_delivery/171207/FILE171207-095406F/10148.jpg";
        let (target_h, target_w) = (256, 256);
        let image = vision::image::load(image_file)?
            .to_device(Device::Cpu)
            .resize2d_letterbox(target_h, target_w)?
            .to_kind(Kind::Float)
            .g_div1(255.0);
        Ok(())
    }
}
