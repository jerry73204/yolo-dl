use anyhow::{Context, Result};
use std::{path::PathBuf, sync::Arc};
use structopt::StructOpt;
use tracing_subscriber::{prelude::*, EnvFilter};
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
    // setup tracing
    {
        let fmt_layer = tracing_subscriber::fmt::layer().with_target(true).compact();
        let filter_layer = EnvFilter::from_default_env();

        tracing_subscriber::registry()
            .with(filter_layer)
            .with(fmt_layer)
            .init();
    }

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
