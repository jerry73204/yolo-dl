use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "Items")]
pub struct Darknet {
    pub net: Net,
    pub layers: Vec<Layer>,
}

impl Darknet {
    pub fn load<P>(config_file: P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        Ok(Self::from_str(&fs::read_to_string(config_file)?)?)
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
            .map_err(|_| format_err!("the first section must be [net]"))?;

        // build layers
        let layers: Vec<_> = items_iter
            .map(|item| item.try_into_layer_config())
            .try_collect()?;

        Ok(Self { net, layers })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Layer {
    Connected(Connected),
    Convolutional(Convolutional),
    Route(Route),
    Shortcut(Shortcut),
    MaxPool(MaxPool),
    UpSample(UpSample),
    BatchNorm(BatchNorm),
    Dropout(Dropout),
    Softmax(Softmax),
    Cost(Cost),
    Crop(Crop),
    AvgPool(AvgPool),
    Yolo(Yolo),
    GaussianYolo(GaussianYolo),
    Unimplemented(UnimplementedLayer),
}

impl Layer {
    pub fn common(&self) -> &Common {
        match self {
            Layer::Connected(layer) => &layer.common,
            Layer::Convolutional(layer) => &layer.common,
            Layer::Route(layer) => &layer.common,
            Layer::Shortcut(layer) => &layer.common,
            Layer::MaxPool(layer) => &layer.common,
            Layer::UpSample(layer) => &layer.common,
            Layer::BatchNorm(layer) => &layer.common,
            Layer::Dropout(layer) => &layer.common,
            Layer::Softmax(layer) => &layer.common,
            Layer::Cost(layer) => &layer.common,
            Layer::Crop(layer) => &layer.common,
            Layer::AvgPool(layer) => &layer.common,
            Layer::Yolo(layer) => &layer.common,
            Layer::GaussianYolo(layer) => &layer.common,
            Layer::Unimplemented(_layer) => panic!("unimplemented layer"),
        }
    }
}
