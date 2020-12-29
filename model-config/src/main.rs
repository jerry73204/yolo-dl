use anyhow::Result;
use model_config::{config::Model, graph::Graph};
use std::{
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
};
use structopt::StructOpt;

fn main() -> Result<()> {
    #[derive(Debug, Clone, StructOpt)]
    enum Args {
        Info {
            /// configuration file
            config_file: PathBuf,
        },
        MakeDotFile {
            /// configuration file
            config_file: PathBuf,
            /// output DOT file
            output_file: PathBuf,
        },
    }

    match Args::from_args() {
        Args::Info { config_file } => {
            info(config_file)?;
        }
        Args::MakeDotFile {
            config_file,
            output_file,
        } => {
            make_dot_file(config_file, output_file)?;
        }
    }

    Ok(())
}

fn info(config_file: impl AsRef<Path>) -> Result<()> {
    let _config = Model::load(config_file)?;
    Ok(())
}

#[cfg(feature = "dot")]
fn make_dot_file(config_file: impl AsRef<Path>, output_file: impl AsRef<Path>) -> Result<()> {
    let config = Model::load(config_file)?;
    let graph = Graph::new(&config)?;
    let mut writer = BufWriter::new(File::create(output_file)?);
    graph.render_dot(&mut writer)?;
    Ok(())
}

#[cfg(not(feature = "dot"))]
fn make_dot_file(_config_file: impl AsRef<Path>, _output_file: impl AsRef<Path>) -> Result<()> {
    use anyhow::bail;
    bail!("'dot' feature must be enabled to run this command");
}
