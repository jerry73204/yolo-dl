use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UpSample {
    #[serde(default = "defaults::upsample_stride")]
    pub stride: usize,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub reverse: bool,
    #[serde(flatten)]
    pub common: Common,
}

impl UpSample {
    pub fn output_shape(&self, input_shape: [usize; 3]) -> [usize; 3] {
        let Self {
            stride, reverse, ..
        } = *self;
        let [in_h, in_w, in_c] = input_shape;
        let (out_h, out_w) = if reverse {
            (in_h / stride, in_w / stride)
        } else {
            (in_h * stride, in_w * stride)
        };
        let out_c = in_c;
        [out_h, out_w, out_c]
    }
}
