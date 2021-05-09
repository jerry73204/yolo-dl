use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "RawCommon")]
pub struct Common {
    pub clip: Option<R64>,
    #[serde(
        rename = "onlyforward",
        with = "serde_::zero_one_bool",
        default = "defaults::bool_false"
    )]
    pub only_forward: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub dont_update: bool,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub burnin_update: bool,
    #[serde(rename = "stopbackward", default = "defaults::stop_backward")]
    pub stop_backward: u64,
    #[serde(with = "serde_::zero_one_bool", default = "defaults::bool_false")]
    pub train_only_bn: bool,
    #[serde(
        rename = "dontload",
        with = "serde_::zero_one_bool",
        default = "defaults::bool_false"
    )]
    pub dont_load: bool,
    #[serde(
        rename = "dontloadscales",
        with = "serde_::zero_one_bool",
        default = "defaults::bool_false"
    )]
    pub dont_load_scales: bool,
    #[serde(rename = "learning_rate", default = "defaults::learning_rate_scale")]
    pub learning_rate_scale: R64,
}

impl TryFrom<RawCommon> for Common {
    type Error = Error;

    fn try_from(from: RawCommon) -> Result<Self, Self::Error> {
        let RawCommon {
            clip,
            only_forward,
            dont_update,
            burnin_update,
            stop_backward,
            train_only_bn,
            dont_load,
            dont_load_scales,
            learning_rate_scale,
        } = from;

        let parse_r64 = |text: &str| -> Result<R64> {
            R64::try_new(f64::from_str(text)?)
                .ok_or_else(|| format_err!("'{}' is not a finite number", text))
        };

        let parse_zero_one_bool = |text: &str| -> Result<_> {
            let value = match text.trim() {
                "0" => false,
                "1" => true,
                _ => bail!("expect 0 or 1, but get '{}'", text),
            };
            Ok(value)
        };

        let clip = clip.map(|clip| parse_r64(clip.as_ref())).transpose()?;

        let only_forward = only_forward
            .map(|text| parse_zero_one_bool(text.as_ref()))
            .transpose()?
            .unwrap_or(false);

        let dont_update = dont_update
            .map(|text| parse_zero_one_bool(text.as_ref()))
            .transpose()?
            .unwrap_or(false);

        let burnin_update = burnin_update
            .map(|text| parse_zero_one_bool(text.as_ref()))
            .transpose()?
            .unwrap_or(false);

        let stop_backward = stop_backward
            .map(|text| text.parse())
            .transpose()?
            .unwrap_or_else(defaults::stop_backward);

        let train_only_bn = train_only_bn
            .map(|text| parse_zero_one_bool(text.as_ref()))
            .transpose()?
            .unwrap_or(false);

        let dont_load = dont_load
            .map(|text| parse_zero_one_bool(text.as_ref()))
            .transpose()?
            .unwrap_or(false);

        let dont_load_scales = dont_load_scales
            .map(|text| parse_zero_one_bool(text.as_ref()))
            .transpose()?
            .unwrap_or(false);

        let learning_rate_scale = learning_rate_scale
            .map(|text| parse_r64(text.as_ref()))
            .transpose()?
            .unwrap_or_else(defaults::learning_rate_scale);

        Ok(Self {
            clip,
            only_forward,
            dont_update,
            burnin_update,
            stop_backward,
            train_only_bn,
            dont_load,
            dont_load_scales,
            learning_rate_scale,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct RawCommon {
    pub clip: Option<String>,
    #[serde(rename = "onlyforward")]
    pub only_forward: Option<String>,
    pub dont_update: Option<String>,
    pub burnin_update: Option<String>,
    #[serde(rename = "stopbackward")]
    pub stop_backward: Option<String>,
    pub train_only_bn: Option<String>,
    #[serde(rename = "dontload")]
    pub dont_load: Option<String>,
    #[serde(rename = "dontloadscales")]
    pub dont_load_scales: Option<String>,
    #[serde(rename = "learning_rate")]
    pub learning_rate_scale: Option<String>,
}
