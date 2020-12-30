mod config;
mod helper;
mod misc;
mod model;
mod module;

pub use config::*;
pub use helper::*;
pub use misc::*;
pub use model::*;
pub use module::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::*;

    #[test]
    fn yolo_v5_small_serde_test() -> Result<()> {
        let init = yolo_v5_small_init(3, 80);
        let text = serde_json::to_string_pretty(&init)?;
        let recovered = serde_json::from_str(&text)?;
        assert_eq!(init, recovered);
        Ok(())
    }
}
