use anyhow::{Context, Result};
use std::{env, path::PathBuf, sync::Arc};
use structopt::StructOpt;
use train::config::Config;

#[derive(Debug, Clone, StructOpt)]
/// Train YOLO model
struct Args {
    #[structopt(long, default_value = "train.json5")]
    /// configuration file
    pub config_file: PathBuf,
}

#[tokio::main]
pub async fn main() -> Result<()> {
    // parse arguments
    let Args { config_file } = Args::from_args();
    let config = Arc::new(
        Config::open(&config_file)
            .with_context(|| format!("failed to load config file '{}'", config_file.display()))?,
    );

    // start training program
    train::start(config).await?;

    Ok(())
}
