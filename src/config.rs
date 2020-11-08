use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "Vec<Item>")]
#[serde(into = "Vec<Item>")]
pub struct Config {
    pub net: Net,
    pub layers: Vec<Layer>,
}

impl TryFrom<Vec<Item>> for Config {
    type Error = anyhow::Error;

    fn try_from(items: Vec<Item>) -> Result<Self, Self::Error> {
        let mut items = items.into_iter();
        let net = {
            let first = items.next().ok_or_else(|| format_err!("no items found"))?;
            match first {
                Item::Net(net) => net,
                _ => bail!("the first item must be 'net'"),
            }
        };
        let layers: Vec<_> = items
            .map(|item| {
                let layer = match item {
                    Item::Convolutional(layer) => Layer::Convolutional(layer),
                    Item::Route(layer) => Layer::Route(layer),
                    Item::Shortcut(layer) => Layer::Shortcut(layer),
                    Item::MaxPool(layer) => Layer::MaxPool(layer),
                    Item::UpSample(layer) => Layer::UpSample(layer),
                    Item::Yolo(layer) => Layer::Yolo(layer),
                    Item::Net(_layer) => bail!("the 'net' layer must appear in the first section"),
                };
                Ok(layer)
            })
            .try_collect()?;

        Ok(Config { net, layers })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Layer {
    #[serde(rename = "convolutional")]
    Convolutional(Convolutional),
    #[serde(rename = "route")]
    Route(Route),
    #[serde(rename = "shortcut")]
    Shortcut(Shortcut),
    #[serde(rename = "maxpool")]
    MaxPool(MaxPool),
    #[serde(rename = "upsample")]
    UpSample(UpSample),
    #[serde(rename = "yolo")]
    Yolo(Yolo),
}

impl From<Config> for Vec<Item> {
    fn from(config: Config) -> Self {
        let Config { net, layers } = config;
        let items: Vec<_> = iter::once(Item::Net(net))
            .chain(layers.into_iter().map(|layer| match layer {
                Layer::Convolutional(layer) => Item::Convolutional(layer),
                Layer::Route(layer) => Item::Route(layer),
                Layer::Shortcut(layer) => Item::Shortcut(layer),
                Layer::MaxPool(layer) => Item::MaxPool(layer),
                Layer::UpSample(layer) => Item::UpSample(layer),
                Layer::Yolo(layer) => Item::Yolo(layer),
            }))
            .collect();
        items
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Item {
    #[serde(rename = "net")]
    Net(Net),
    #[serde(rename = "convolutional")]
    Convolutional(Convolutional),
    #[serde(rename = "route")]
    Route(Route),
    #[serde(rename = "shortcut")]
    Shortcut(Shortcut),
    #[serde(rename = "maxpool")]
    MaxPool(MaxPool),
    #[serde(rename = "upsample")]
    UpSample(UpSample),
    #[serde(rename = "yolo")]
    Yolo(Yolo),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Net {
    pub batch: usize,
    pub subdivisions: usize,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub momentum: R64,
    pub decay: R64,
    pub angle: R64,
    pub saturation: R64,
    pub exposure: R64,
    pub hue: R64,
    pub learning_rate: R64,
    pub burn_in: usize,
    pub max_batches: usize,
    pub policy: Policy,
    #[serde(with = "serde_opt_vec_usize")]
    pub steps: Option<Vec<usize>>,
    #[serde(with = "serde_opt_vec_r64")]
    pub scales: Option<Vec<R64>>,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub cutmux: bool,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub mosaic: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Convolutional {
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub batch_normalize: bool,
    pub filters: usize,
    pub size: usize,
    // pub stride_x: usize,
    // pub stride_y: usize,
    #[serde(default = "default_stride")]
    pub stride: usize,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub pad: bool,
    // pub padding: usize,
    #[serde(default = "default_groups")]
    pub groups: usize,
    #[serde(default = "default_dilation")]
    pub dilation: usize,
    pub antialiasing: usize,
    pub share_index: Option<usize>,
    pub activation: Activation,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Route {
    #[serde(with = "serde_vec_isize")]
    pub layers: Vec<isize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shortcut {
    pub from: isize,
    pub activation: Activation,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MaxPool {
    pub stride: usize,
    pub size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UpSample {
    pub stride: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Yolo {
    #[serde(with = "serde_vec_usize")]
    pub mask: Vec<usize>,
    #[serde(with = "serde_anchors")]
    pub anchors: Vec<(usize, usize)>,
    pub classes: usize,
    pub num: usize,
    pub jitter: R64,
    pub ignore_thresh: R64,
    pub truth_thresh: R64,
    #[serde(with = "serde_zero_one_bool", default = "default_bool_false")]
    pub random: bool,
    pub scale_x_y: R64,
    pub iou_thresh: R64,
    pub cls_normalizer: R64,
    pub iou_normalizer: R64,
    pub iou_loss: IouLoss,
    pub nms_kind: NmsKind,
    pub beta_nms: R64,
    pub max_delta: R64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Activation {
    #[serde(rename = "mish")]
    Mish,
    #[serde(rename = "hard_mish")]
    HardMish,
    #[serde(rename = "swish")]
    Swish,
    #[serde(rename = "normalize_channels")]
    NormalizeChannels,
    #[serde(rename = "normalize_channels_softmax")]
    NormalizeChannelsSoftmax,
    #[serde(rename = "normalize_channels_softmax_maxval")]
    NormalizeChannelsSoftmaxMaxval,
    #[serde(rename = "logistic")]
    Logistic,
    #[serde(rename = "loggy")]
    Loggy,
    #[serde(rename = "relu")]
    Relu,
    #[serde(rename = "elu")]
    Elu,
    #[serde(rename = "selu")]
    Selu,
    #[serde(rename = "gelu")]
    Gelu,
    #[serde(rename = "relie")]
    Relie,
    #[serde(rename = "ramp")]
    Ramp,
    #[serde(rename = "linear")]
    Linear,
    #[serde(rename = "tanh")]
    Tanh,
    #[serde(rename = "plse")]
    Plse,
    #[serde(rename = "leaky")]
    Leaky,
    #[serde(rename = "stair")]
    Stair,
    #[serde(rename = "hardtan")]
    Hardtan,
    #[serde(rename = "lhtan")]
    Lhtan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IouLoss {
    #[serde(rename = "mse")]
    Mse,
    #[serde(rename = "iou")]
    IoU,
    #[serde(rename = "giou")]
    GIoU,
    #[serde(rename = "diou")]
    DIoU,
    #[serde(rename = "ciou")]
    CIoU,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NmsKind {
    #[serde(rename = "default")]
    Default,
    #[serde(rename = "greedynms")]
    Greedy,
    #[serde(rename = "diounms")]
    DIoU,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Policy {
    #[serde(rename = "random")]
    Random,
    #[serde(rename = "poly")]
    Poly,
    #[serde(rename = "constant")]
    Constant,
    #[serde(rename = "step")]
    Step,
    #[serde(rename = "exp")]
    Exp,
    #[serde(rename = "sigmoid")]
    Sigmoid,
    #[serde(rename = "steps")]
    Steps,
    #[serde(rename = "sgdr")]
    Sgdr,
}

// utility functions

fn default_bool_false() -> bool {
    false
}

fn default_groups() -> usize {
    1
}

fn default_stride() -> usize {
    1
}

fn default_dilation() -> usize {
    1
}

mod serde_zero_one_bool {
    use super::*;

    pub fn serialize<S>(&yes: &bool, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (yes as i64).serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<bool, D::Error>
    where
        D: Deserializer<'de>,
    {
        match i64::deserialize(deserializer)? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(D::Error::invalid_value(
                de::Unexpected::Signed(value),
                &"0 or 1",
            )),
        }
    }
}

mod serde_vec_isize {
    use super::*;

    pub fn serialize<S>(steps: &Vec<isize>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        steps
            .iter()
            .map(|step| step.to_string())
            .join(",")
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<isize>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        let steps: Vec<isize> = text
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            .split(",")
            .map(|token| token.parse())
            .try_collect()
            .map_err(|err| D::Error::custom(format!("failed to parse steps: {:?}", err)))?;
        Ok(steps)
    }
}

mod serde_vec_usize {
    use super::*;

    pub fn serialize<S>(steps: &Vec<usize>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        steps
            .iter()
            .map(|step| step.to_string())
            .join(",")
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<usize>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        let steps: Vec<usize> = text
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            .split(",")
            .map(|token| token.parse())
            .try_collect()
            .map_err(|err| D::Error::custom(format!("failed to parse steps: {:?}", err)))?;
        Ok(steps)
    }
}

mod serde_opt_vec_usize {
    use super::*;

    pub fn serialize<S>(steps: &Option<Vec<usize>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        steps
            .as_ref()
            .map(|steps| steps.iter().map(|step| step.to_string()).join(","))
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Vec<usize>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = <Option<String>>::deserialize(deserializer)?;
        let steps: Option<Vec<usize>> = text
            .map(|text| {
                text.chars()
                    .filter(|c| !c.is_whitespace())
                    .collect::<String>()
                    .split(",")
                    .map(|token| token.parse())
                    .try_collect()
            })
            .transpose()
            .map_err(|err| D::Error::custom(format!("failed to parse steps: {:?}", err)))?;
        Ok(steps)
    }
}

mod serde_opt_vec_r64 {
    use super::*;

    pub fn serialize<S>(scales: &Option<Vec<R64>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        scales
            .as_ref()
            .map(|steps| steps.iter().map(|step| step.to_string()).join(","))
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Vec<R64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = <Option<String>>::deserialize(deserializer)?;
        let scales: Option<Vec<R64>> = text
            .map(|text| {
                text.chars()
                    .filter(|c| !c.is_whitespace())
                    .collect::<String>()
                    .split(",")
                    .map(|token| {
                        let value: f64 = token.parse().map_err(|err| {
                            D::Error::custom(format!("failed to parse steps: {:?}", err))
                        })?;
                        let value = R64::try_new(value).ok_or_else(|| {
                            D::Error::custom(format!("invalid value '{}'", token))
                        })?;
                        Ok(value)
                    })
                    .try_collect()
            })
            .transpose()?;
        Ok(scales)
    }
}

mod serde_anchors {
    use super::*;

    pub fn serialize<S>(steps: &Vec<(usize, usize)>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        steps
            .iter()
            .flat_map(|(w, h)| vec![w, h])
            .map(|val| val.to_string())
            .join(",")
            .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<(usize, usize)>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        let values: Vec<usize> = text
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            .split(",")
            .map(|token| token.parse())
            .try_collect()
            .map_err(|err| D::Error::custom(format!("failed to parse anchors: {:?}", err)))?;

        if values.len() % 2 != 0 {
            return Err(D::Error::custom("expect even number of values"));
        }

        let anchors: Vec<_> = values
            .into_iter()
            .chunks(2)
            .into_iter()
            .map(|mut chunk| (chunk.next().unwrap(), chunk.next().unwrap()))
            .collect();
        Ok(anchors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wtf() -> Result<()> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("yolov4.cfg");
        let config: Config = serde_ini::from_str(&fs::read_to_string(path)?)?;
        dbg!(config);
        Ok(())
    }
}
