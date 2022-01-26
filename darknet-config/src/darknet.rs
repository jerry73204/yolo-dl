use super::{
    AvgPool, BatchNorm, Connected, Convolutional, Cost, Crop, Dropout, GaussianYolo, Layer,
    MaxPool, Net, Route, Shortcut, Softmax, UnimplementedLayer, UpSample, Yolo,
};
use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize)]
#[serde(try_from = "Items")]
pub struct Darknet {
    pub net: Net,
    pub layers: Vec<Layer>,
}

impl Darknet {
    pub fn load<P>(file: P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        Self::from_str(&fs::read_to_string(file)?)
    }

    pub fn to_string(&self) -> Result<String> {
        Ok(serde_ini::to_string(self)?)
    }
}

impl FromStr for Darknet {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        // remove comments and trailing whitespaces
        let regex = RegexBuilder::new(r" *(#.*)?$")
            .multi_line(true)
            .build()
            .unwrap();
        let text = regex.replace_all(text, "");

        // parse
        Ok(serde_ini::from_str(&text)?)
    }
}

impl TryFrom<Items> for Darknet {
    type Error = anyhow::Error;

    fn try_from(Items(items): Items) -> Result<Self, Self::Error> {
        // ensure only the first item is "net" item
        {
            let mut iter = items.iter();
            ensure!(
                matches!(iter.next(), Some(Item::Net(_))),
                "the first item must be [net]"
            );
            ensure!(
                iter.all(|item| !matches!(item, Item::Net(_))),
                "net item must be the first item"
            );
        };

        let mut items_iter = items.into_iter();

        // build net item
        let net = items_iter
            .next()
            .unwrap()
            .try_into_net()
            .map_err(|_| anyhow!("the first section must be [net]"))?;

        // build layers
        let layers: Vec<_> = items_iter
            .map(|item| item.try_into_layer_config())
            .try_collect()?;

        Ok(Self { net, layers })
    }
}

impl Serialize for Darknet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Items::try_from(self.clone())
            .map_err(|err| {
                S::Error::custom(format!(
                    "unable to serialize darknet configuration: {:?}",
                    err
                ))
            })?
            .serialize(serializer)
    }
}

pub(crate) use item::*;
mod item {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub(crate) enum Item {
        #[serde(rename = "net")]
        Net(Net),
        #[serde(rename = "connected")]
        Connected(Connected),
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
        #[serde(rename = "batchnorm")]
        BatchNorm(BatchNorm),
        #[serde(rename = "dropout")]
        Dropout(Dropout),
        #[serde(rename = "softmax")]
        Softmax(Softmax),
        #[serde(rename = "Gaussian_yolo")]
        GaussianYolo(GaussianYolo),
        #[serde(rename = "yolo")]
        Yolo(Yolo),
        #[serde(rename = "cost")]
        Cost(Cost),
        #[serde(rename = "crop")]
        Crop(Crop),
        #[serde(rename = "avgpool")]
        AvgPool(AvgPool),
        #[serde(rename = "local_avgpool")]
        LocalAvgPool(UnimplementedLayer),
        #[serde(rename = "crnn")]
        Crnn(UnimplementedLayer),
        #[serde(rename = "sam")]
        Sam(UnimplementedLayer),
        #[serde(rename = "scale_channels")]
        ScaleChannels(UnimplementedLayer),
        #[serde(rename = "gru")]
        Gru(UnimplementedLayer),
        #[serde(rename = "lstm")]
        Lstm(UnimplementedLayer),
        #[serde(rename = "rnn")]
        Rnn(UnimplementedLayer),
        #[serde(rename = "detection")]
        Detection(UnimplementedLayer),
        #[serde(rename = "region")]
        Region(UnimplementedLayer),
        #[serde(rename = "reorg")]
        Reorg(UnimplementedLayer),
        #[serde(rename = "contrastive")]
        Contrastive(UnimplementedLayer),
    }

    impl Item {
        pub fn try_into_net(self) -> Result<Net> {
            let net = match self {
                Self::Net(net) => net,
                _ => bail!("not a net layer"),
            };
            Ok(net)
        }

        pub fn try_into_layer_config(self) -> Result<Layer> {
            let layer = match self {
                Self::Connected(layer) => Layer::Connected(layer),
                Self::Convolutional(layer) => Layer::Convolutional(layer),
                Self::Route(layer) => Layer::Route(layer),
                Self::Shortcut(layer) => Layer::Shortcut(layer),
                Self::MaxPool(layer) => Layer::MaxPool(layer),
                Self::UpSample(layer) => Layer::UpSample(layer),
                Self::BatchNorm(layer) => Layer::BatchNorm(layer),
                Self::Dropout(layer) => Layer::Dropout(layer),
                Self::Softmax(layer) => Layer::Softmax(layer),
                Self::Cost(layer) => Layer::Cost(layer),
                Self::Crop(layer) => Layer::Crop(layer),
                Self::AvgPool(layer) => Layer::AvgPool(layer),
                Self::GaussianYolo(layer) => Layer::GaussianYolo(layer),
                Self::Yolo(layer) => Layer::Yolo(layer),
                Self::Net(_layer) => {
                    bail!("the 'net' layer must appear in the first section")
                }
                // unimplemented
                Self::Crnn(layer)
                | Self::Sam(layer)
                | Self::ScaleChannels(layer)
                | Self::LocalAvgPool(layer)
                | Self::Contrastive(layer)
                | Self::Detection(layer)
                | Self::Region(layer)
                | Self::Reorg(layer)
                | Self::Rnn(layer)
                | Self::Lstm(layer)
                | Self::Gru(layer) => Layer::Unimplemented(layer),
            };
            Ok(layer)
        }
    }

    impl TryFrom<Darknet> for Items {
        type Error = Error;

        fn try_from(config: Darknet) -> Result<Self, Self::Error> {
            let Darknet {
                net,
                layers: orig_layers,
            } = config;

            let items: Vec<_> = {
                chain!([Ok(Item::Net(net))], {
                    orig_layers.into_iter().map(|layer| -> Result<_> {
                        let item = match layer {
                            Layer::Connected(layer) => Item::Connected(layer),
                            Layer::Convolutional(layer) => Item::Convolutional(layer),
                            Layer::Route(layer) => Item::Route(layer),
                            Layer::Shortcut(layer) => Item::Shortcut(layer),
                            Layer::MaxPool(layer) => Item::MaxPool(layer),
                            Layer::UpSample(layer) => Item::UpSample(layer),
                            Layer::BatchNorm(layer) => Item::BatchNorm(layer),
                            Layer::Dropout(layer) => Item::Dropout(layer),
                            Layer::Softmax(layer) => Item::Softmax(layer),
                            Layer::Cost(layer) => Item::Cost(layer),
                            Layer::Crop(layer) => Item::Crop(layer),
                            Layer::AvgPool(layer) => Item::AvgPool(layer),
                            Layer::Unimplemented(_layer) => bail!("unimplemented layer"),
                            Layer::Yolo(layer) => Item::Yolo(layer),
                            Layer::GaussianYolo(layer) => Item::GaussianYolo(layer),
                        };

                        Ok(item)
                    })
                })
                .try_collect()?
            };

            Ok(Items(items))
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[serde(transparent)]
    pub(crate) struct Items(pub Vec<Item>);
}
