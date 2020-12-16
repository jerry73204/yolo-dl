use anyhow::Result;
use model_config::Config;
use std::{
    fs,
    path::{Path, PathBuf},
};
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
    let _config: Config = json5::from_str(&fs::read_to_string(config_file)?)?;
    Ok(())
}
