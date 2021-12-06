use anyhow::{Context, Result};
use darknet_config::Darknet;
use std::path::Path;

#[test]
fn load_darknet_config() -> Result<()> {
    glob::glob(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("cfg")
            .join("*.cfg")
            .to_str()
            .unwrap(),
    )?
    .try_for_each(|path| -> Result<_> {
        let path = path?;
        let _config =
            Darknet::load(&path).with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(())
    })?;

    Ok(())
}

// use serde::Serialize;
// use serde::Deserialize;
// use anyhow::Error;
// use anyhow::bail;

// #[derive(Clone, Serialize, Deserialize)]
// struct Outer {
//     #[serde(flatten)]
//     inner: Inner,
//     #[serde(flatten)]
//     inner2: Inner2,
// }

// #[derive(Clone, Serialize, Deserialize)]
// #[serde(try_from = "RawInner", into = "RawInner")]
// enum Inner {
//     A,
//     B,
// }

// #[derive(Clone, Serialize, Deserialize)]
// struct RawInner {
//     pub a: bool,
//     pub b: bool,
// }

// impl From<Inner> for RawInner {
//     fn from(from: Inner) -> Self {
//         let (a, b) = match from {
//             Inner::A => (true, false),
//             Inner::B => (false, true)
//         };
//         Self {a, b}
//     }
// }

// impl TryFrom<RawInner> for Inner {
//     type Error = Error;

//     fn try_from(from: RawInner) -> Result<Self, Self::Error> {
//         let RawInner {a, b} = from;
//         let value = match (a, b) {
//             (true, false) => Self::A,
//             (false, true) => Self::B,
//             _ => bail!("error")
//         };
//         Ok(value)
//     }
// }

// #[derive(Clone, Serialize, Deserialize)]
// struct Inner2 {
//     pub name: Option<String>,
// }

// #[test]
// fn wtf() {
//     let _: Outer = serde_json::from_str(r#"{"a": true, "b": false}"#).unwrap();
// }
