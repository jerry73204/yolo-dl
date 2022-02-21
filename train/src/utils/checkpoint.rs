use crate::{common::*, config::LoadCheckpoint};
use regex::Regex;

pub const FILE_STRFTIME: &str = "%Y-%m-%d-%H-%M-%S.%3f%z";

/// Save parameters to a checkpoint file.
pub fn save_checkpoint(
    vs: &nn::VarStore,
    checkpoint_dir: &Path,
    training_step: usize,
    loss: f64,
) -> Result<()> {
    let filename = format!(
        "{}_{:06}_{:08.5}.ckpt",
        Local::now().format(FILE_STRFTIME),
        training_step,
        loss
    );
    let path = checkpoint_dir.join(filename);
    vs.save(&path)?;
    Ok(())
}

/// Load parameters from a diretory with specified checkpoint loading method.
pub fn try_load_checkpoint(
    vs: &mut nn::VarStore,
    logging_dir: &Path,
    load_checkpoint: &LoadCheckpoint,
) -> Result<()> {
    let checkpoint_filename_regex =
        Regex::new(r"^(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.\d{3}\+\d{4})_\d{6}_\d+\.\d+\.ckpt$")
            .unwrap();

    let path = match load_checkpoint {
        LoadCheckpoint::Disabled => {
            info!("checkpoint loading is disabled");
            None
        }
        LoadCheckpoint::FromRecent => {
            let paths: Vec<_> =
                glob::glob(&format!("{}/*/checkpoints/*.ckpt", logging_dir.display()))
                    .unwrap()
                    .try_collect()?;
            let paths = paths
                .into_iter()
                .filter_map(|path| {
                    let file_name = path.file_name()?.to_str()?;
                    let captures = checkpoint_filename_regex.captures(file_name)?;
                    let datetime_str = captures.get(1)?.as_str();
                    let datetime = DateTime::parse_from_str(datetime_str, FILE_STRFTIME).unwrap();
                    Some((path, datetime))
                })
                .collect_vec();
            let checkpoint_file = paths
                .into_iter()
                .max_by_key(|(_path, datetime)| datetime.clone())
                .map(|(path, _datetime)| path);

            if let None = &checkpoint_file {
                warn!("no checkpoint file found");
            }

            checkpoint_file
        }
        LoadCheckpoint::FromFile { file } => {
            if file.is_file() {
                Some(file.to_owned())
            } else {
                warn!("{} is not a file", file.display());
                None
            }
        }
    };

    if let Some(path) = path {
        info!("load checkpoint file {}", path.display());
        vs.load_partial(path)?;
    }

    Ok(())
}
