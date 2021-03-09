mod common;
pub mod config;
pub mod input_stream;

use crate::{common::*, config::Config};

pub async fn start(_config: Arc<Config>) -> Result<()> {
    Ok(())
}
