use super::*;

pub use darknet::*;

mod darknet {
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
            Self::from_str(&fs::read_to_string(config_file)?)
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
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, strum::AsRefStr)]
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

impl From<UnimplementedLayer> for Layer {
    fn from(v: UnimplementedLayer) -> Self {
        Self::Unimplemented(v)
    }
}

impl From<GaussianYolo> for Layer {
    fn from(v: GaussianYolo) -> Self {
        Self::GaussianYolo(v)
    }
}

impl From<Yolo> for Layer {
    fn from(v: Yolo) -> Self {
        Self::Yolo(v)
    }
}

impl From<AvgPool> for Layer {
    fn from(v: AvgPool) -> Self {
        Self::AvgPool(v)
    }
}

impl From<Crop> for Layer {
    fn from(v: Crop) -> Self {
        Self::Crop(v)
    }
}

impl From<Cost> for Layer {
    fn from(v: Cost) -> Self {
        Self::Cost(v)
    }
}

impl From<Softmax> for Layer {
    fn from(v: Softmax) -> Self {
        Self::Softmax(v)
    }
}

impl From<Dropout> for Layer {
    fn from(v: Dropout) -> Self {
        Self::Dropout(v)
    }
}

impl From<BatchNorm> for Layer {
    fn from(v: BatchNorm) -> Self {
        Self::BatchNorm(v)
    }
}

impl From<UpSample> for Layer {
    fn from(v: UpSample) -> Self {
        Self::UpSample(v)
    }
}

impl From<Shortcut> for Layer {
    fn from(v: Shortcut) -> Self {
        Self::Shortcut(v)
    }
}

impl From<Route> for Layer {
    fn from(v: Route) -> Self {
        Self::Route(v)
    }
}

impl From<Convolutional> for Layer {
    fn from(v: Convolutional) -> Self {
        Self::Convolutional(v)
    }
}

impl From<Connected> for Layer {
    fn from(v: Connected) -> Self {
        Self::Connected(v)
    }
}

impl Layer {
    pub fn output_shape(&self, input_shape: &InputShape) -> Option<OutputShape> {
        let output_shape: OutputShape = match self {
            Layer::Connected(layer) => layer.output_shape(input_shape.single_dim1()?)?.into(),
            Layer::Convolutional(layer) => layer.output_shape(input_shape.single_dim3()?).into(),
            Layer::Route(layer) => layer.output_shape(&input_shape.multiple_dim3()?)?.into(),
            Layer::Shortcut(layer) => layer.output_shape(&input_shape.multiple_dim3()?)?.into(),
            Layer::MaxPool(layer) => layer.output_shape(input_shape.single_dim3()?).into(),
            Layer::UpSample(layer) => layer.output_shape(input_shape.single_dim3()?).into(),
            Layer::BatchNorm(layer) => layer.output_shape(input_shape.single()?).into(),
            Layer::Dropout(layer) => layer.output_shape(input_shape.single()?).into(),
            Layer::Softmax(layer) => layer.output_shape(input_shape.single()?).into(),
            Layer::Yolo(layer) => layer.output_shape(input_shape.single_dim3()?)?,
            Layer::GaussianYolo(layer) => layer.output_shape(input_shape.single_dim3()?)?,
            Layer::Cost(_layer) => unimplemented!(),
            Layer::Crop(_layer) => unimplemented!(),
            Layer::AvgPool(_layer) => unimplemented!(),
            Layer::Unimplemented(_layer) => return None,
        };
        Some(output_shape)
    }

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
