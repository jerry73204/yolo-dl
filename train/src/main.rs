use anyhow::{Context, Result};
use std::{env, path::PathBuf, sync::Arc};
use structopt::StructOpt;
use tracing::{trace_span, Instrument};
use tracing_subscriber::{filter::LevelFilter, prelude::*, EnvFilter};
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
    let fmt_layer = tracing_subscriber::fmt::layer().with_target(true).compact();
    let filter_layer = {
        let filter = EnvFilter::from_default_env();
        let filter = if let Err(_) = env::var("RUST_LOG") {
            filter.add_directive(LevelFilter::INFO.into())
        } else {
            filter
        };
        filter
    };

    let tracer = opentelemetry_jaeger::new_pipeline()
        .with_service_name("train")
        .install_simple()?;
    let otel_layer = tracing_opentelemetry::OpenTelemetryLayer::new(tracer);

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .with(otel_layer)
        .init();

    // parse arguments
    let Args { config_file } = Args::from_args();
    let config = Arc::new(
        Config::open(&config_file)
            .with_context(|| format!("failed to load config file '{}'", config_file.display()))?,
    );

    // start training program
    train::start(config)
        .instrument(trace_span!("train"))
        .await?;

    Ok(())
}
