use anyhow::Result;
use model_graph::Graph;
use prettytable::{cell, row, Table};
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
    let graph = Graph::load_newslab_v1_json(config_file)?;
    let nodes = graph.nodes();

    // print layer information
    {
        let mut table = Table::new();
        table.add_row(row!["key", "kind", "path", "input_keys", "output shape"]);

        nodes.iter().for_each(|(&key, node)| {
            table.add_row(row![
                key,
                node.config.as_ref(),
                node.path
                    .as_ref()
                    .map(|path| format!("{}", path))
                    .unwrap_or(format!("")),
                format!("{:?}", node.input_keys),
                format!("{:?}", node.output_shape),
            ]);
        });

        table.printstd();
    }

    Ok(())
}

#[cfg(feature = "dot")]
fn make_dot_file(config_file: impl AsRef<Path>, output_file: impl AsRef<Path>) -> Result<()> {
    let graph = Graph::load_newslab_v1_json(config_file)?;
    let mut writer = BufWriter::new(File::create(output_file)?);
    graph.render_dot(&mut writer)?;
    Ok(())
}

#[cfg(not(feature = "dot"))]
fn make_dot_file(_config_file: impl AsRef<Path>, _output_file: impl AsRef<Path>) -> Result<()> {
    use anyhow::bail;
    bail!("'dot' feature must be enabled to run this command");
}
