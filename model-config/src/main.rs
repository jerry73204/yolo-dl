use anyhow::Result;
use model_config::config::Model;
use std::path::{Path, PathBuf};
use structopt::StructOpt;

fn main() -> Result<()> {
    #[derive(Debug, Clone, StructOpt)]
    enum Args {
        Info { config_file: PathBuf },
    }

    match Args::from_args() {
        Args::Info { config_file } => {
            info(config_file)?;
        }
    }

    Ok(())
}

fn info(config_file: impl AsRef<Path>) -> Result<()> {
    let _config = Model::load(config_file)?;
    Ok(())
}
