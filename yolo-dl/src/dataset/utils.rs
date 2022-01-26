use crate::common::*;

pub async fn load_classes_file(path: impl AsRef<Path>) -> Result<IndexSet<String>> {
    let path = path.as_ref();
    let content = tokio::fs::read_to_string(path).await?;
    let lines: Vec<_> = content.lines().collect();
    let classes: IndexSet<_> = lines.iter().cloned().map(ToOwned::to_owned).collect();
    ensure!(
        lines.len() == classes.len(),
        "duplicated class names found in '{}'",
        path.display()
    );
    ensure!(
        !classes.is_empty(),
        "no classes found in '{}'",
        path.display()
    );
    Ok(classes)
}
