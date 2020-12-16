mod activation;
mod config;
mod helper;
mod model;
mod module;

pub use activation::*;
pub use config::*;
pub use helper::*;
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
        println!("{}", text);
        let recovered = serde_json::from_str(&text)?;
        assert_eq!(init, recovered);
        Ok(())
    }
}
