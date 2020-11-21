pub mod cache;
mod common;

use crate::common::*;

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

/// Build dataset cache
#[derive(Debug, Clone, FromArgs)]
#[argh(subcommand, name = "build")]
struct BuildArgs {
    /// dataset kind
    #[argh(option)]
    pub kind: String,
    #[argh(positional)]
    pub dataset_dir: PathBuf,
    #[argh(positional)]
    pub output_file: PathBuf,
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
    unimplemented!();
    Ok(())
}

async fn build(args: BuildArgs) -> Result<()> {
    let BuildArgs {
        kind,
        dataset_dir,
        output_file,
    } = args;

    match kind.as_str() {
        "coco" => {}
        "iii" => {
            build_iii_dataset(&dataset_dir, HashSet::new()).await?;
        }
        "voc" => {}
        _ => bail!("dataset kind '{}' is not supported"),
    }

    Ok(())
}

async fn build_iii_dataset(
    dataset_dir: impl AsRef<async_std::path::Path>,
    blacklist_files: HashSet<PathBuf>,
) -> Result<()> {
    // list xml files
    let xml_files = {
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
                .try_collect()?;
            Fallible::Ok(xml_files)
        })
        .await?
    };

    // parse xml files
    let samples: Vec<_> = {
        stream::iter(xml_files.into_iter())
            .par_then(None, move |annotation_file| {
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

                    Fallible::Ok(sample)
                }
            })
            .try_collect()
            .await?
    };

    // build cache meta

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IiiSample {
    pub image_file: PathBuf,
    pub annotation_file: PathBuf,
    pub annotation: voc::Annotation,
}
