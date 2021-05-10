use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "RawConvolutional", into = "RawConvolutional")]
pub struct Convolutional {
    pub filters: usize,
    pub groups: usize,
    pub size: usize,
    pub batch_normalize: bool,
    pub stride_x: usize,
    pub stride_y: usize,
    pub dilation: usize,
    pub antialiasing: bool,
    pub padding: usize,
    pub activation: Activation,
    pub assisted_excitation: bool,
    pub share_index: Option<LayerIndex>,
    pub cbn: bool,
    pub binary: bool,
    pub xnor: bool,
    pub use_bin_output: bool,
    pub deform: Deform,
    pub flipped: bool,
    pub dot: bool,
    pub angle: R64,
    pub grad_centr: bool,
    pub reverse: bool,
    pub coordconv: bool,
    #[serde(flatten)]
    pub common: Common,
}

impl Convolutional {
    pub fn output_shape(&self, [in_h, in_w, _in_c]: [usize; 3]) -> [usize; 3] {
        let Self {
            filters,
            padding,
            size,
            stride_x,
            stride_y,
            ..
        } = *self;
        let out_h = (in_h + 2 * padding - size) / stride_y + 1;
        let out_w = (in_w + 2 * padding - size) / stride_x + 1;
        [out_h, out_w, filters]
    }
}

impl TryFrom<RawConvolutional> for Convolutional {
    type Error = anyhow::Error;

    fn try_from(raw: RawConvolutional) -> Result<Self, Self::Error> {
        let RawConvolutional {
            filters,
            groups,
            size,
            stride,
            stride_x,
            stride_y,
            dilation,
            antialiasing,
            pad,
            padding,
            activation,
            assisted_excitation,
            share_index,
            batch_normalize,
            cbn,
            binary,
            xnor,
            use_bin_output,
            sway,
            rotate,
            stretch,
            stretch_sway,
            flipped,
            dot,
            angle,
            grad_centr,
            reverse,
            coordconv,
            common,
        } = raw;

        let stride_x = stride_x.unwrap_or(stride);
        let stride_y = stride_y.unwrap_or(stride);

        let padding = match (pad, padding) {
            (true, Some(_)) => {
                warn!("padding option is ignored and is set to size / 2 due to pad == 1");
                size / 2
            }
            (true, None) => size / 2,
            (false, padding) => padding.unwrap_or(0),
        };

        let deform = match (sway, rotate, stretch, stretch_sway) {
            (false, false, false, false) => Deform::None,
            (true, false, false, false) => Deform::Sway,
            (false, true, false, false) => Deform::Rotate,
            (false, false, true, false) => Deform::Stretch,
            (false, false, false, true) => Deform::StretchSway,
            _ => bail!("at most one of sway, rotate, stretch, stretch_sway can be set"),
        };

        // sanity check
        let dilation = if size == 1 && dilation != 1 {
            warn!(
                "dilation must be 1 if size is 1, but get dilation = {}, it will be ignored",
                dilation
            );
            1
        } else {
            dilation
        };

        match (deform, size == 1) {
            (Deform::None, _) | (_, false) => (),
            (_, true) => {
                bail!("sway, rotate, stretch, stretch_sway shoud be used with size >= 3")
            }
        }

        ensure!(!xnor || groups == 1, "groups must be 1 if xnor is enabled");

        Ok(Self {
            filters,
            groups,
            size,
            batch_normalize,
            stride_x,
            stride_y,
            dilation,
            antialiasing,
            padding,
            activation,
            assisted_excitation,
            share_index,
            cbn,
            binary,
            xnor,
            use_bin_output,
            deform,
            flipped,
            dot,
            angle,
            grad_centr,
            reverse,
            coordconv,
            common,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct RawConvolutional {
    pub filters: usize,
    #[serde(default = "defaults::groups")]
    pub groups: usize,
    pub size: usize,
    #[serde(default = "defaults::stride")]
    pub stride: usize,
    pub stride_x: Option<usize>,
    pub stride_y: Option<usize>,
    #[serde(default = "defaults::dilation")]
    pub dilation: usize,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub antialiasing: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub pad: bool,
    pub padding: Option<usize>,
    pub activation: Activation,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub assisted_excitation: bool,
    pub share_index: Option<LayerIndex>,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub batch_normalize: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub cbn: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub binary: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub xnor: bool,
    #[serde(
        rename = "bin_output",
        with = "serde_::zero_one_bool",
        default = "defaults::bool_false"
    )]
    pub use_bin_output: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub sway: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub rotate: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub stretch: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub stretch_sway: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub flipped: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub dot: bool,
    #[serde(default = "defaults::angle")]
    pub angle: R64,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub grad_centr: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub reverse: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub coordconv: bool,
    #[serde(flatten)]
    pub common: Common,
}

impl From<Convolutional> for RawConvolutional {
    fn from(conv: Convolutional) -> Self {
        let Convolutional {
            filters,
            groups,
            size,
            stride_x,
            stride_y,
            dilation,
            antialiasing,
            padding,
            activation,
            assisted_excitation,
            share_index,
            batch_normalize,
            cbn,
            binary,
            xnor,
            use_bin_output,
            deform,
            flipped,
            dot,
            angle,
            grad_centr,
            reverse,
            coordconv,
            common,
        } = conv;

        let (sway, rotate, stretch, stretch_sway) = match deform {
            Deform::None => (false, false, false, false),
            Deform::Sway => (true, false, false, false),
            Deform::Rotate => (false, true, false, false),
            Deform::Stretch => (false, false, true, false),
            Deform::StretchSway => (false, false, false, true),
        };

        Self {
            filters,
            groups,
            size,
            stride: defaults::stride(),
            stride_x: Some(stride_x),
            stride_y: Some(stride_y),
            dilation,
            antialiasing,
            pad: false,
            padding: Some(padding),
            activation,
            assisted_excitation,
            share_index,
            batch_normalize,
            cbn,
            binary,
            xnor,
            use_bin_output,
            sway,
            rotate,
            stretch,
            stretch_sway,
            flipped,
            dot,
            angle,
            grad_centr,
            reverse,
            coordconv,
            common,
        }
    }
}
