use crate::common::*;

pub use size::*;
mod size {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(from = "(R64, R64)", into = "(R64, R64)")]
    pub struct Size {
        pub h: R64,
        pub w: R64,
    }

    impl From<(R64, R64)> for Size {
        fn from((h, w): (R64, R64)) -> Self {
            Self { h, w }
        }
    }

    impl From<Size> for (R64, R64) {
        fn from(Size { h, w }: Size) -> Self {
            (h, w)
        }
    }
}
