use crate::common::*;

pub use input::*;
pub use model::*;
pub use output::*;
pub use preprocess::*;

/// The version of configuration format.
///
/// The version number follows the SemVer specification.
pub static CONFIG_VERSION: Lazy<VersionReq> = Lazy::new(|| VersionReq::parse("0.1.0").unwrap());

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(deserialize_with = "deserialize_version")]
    pub version: Version,
    pub model: ModelConfig,
    pub input: InputConfig,
    pub preprocess: PreprocessConfig,
    pub output: OutputConfig,
}

impl Config {
    pub fn open<P>(path: P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let text = fs::read_to_string(path)?;
        let config = json5::from_str(&text)?;
        Ok(config)
    }
}

mod model {
    use super::*;

    /// Model configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ModelConfig {
        pub cfg_file: PathBuf,
        /// Mini-batch size per device.
        pub minibatch_size: NonZeroUsize,
        /// The device where the preprocessor works on.
        #[serde(with = "serde_vec_device")]
        pub devices: Vec<Device>,
    }
}

mod input {
    use super::*;

    /// Dataset options.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InputConfig {
        /// Optional list of whitelisted classes.
        pub class_whitelist: Option<HashSet<String>>,
        /// The dataset configuration.
        pub kind: DatasetKind,
    }

    /// Variants of dataset and options.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum DatasetKind {
        /// Microsoft COCO dataset options.
        Coco {
            classes_file: PathBuf,
            image_size: NonZeroUsize,
            dataset_dir: PathBuf,
            dataset_name: String,
        },
        /// PASCAL VOC dataset options.
        Voc {
            classes_file: PathBuf,
            image_size: NonZeroUsize,
            dataset_dir: PathBuf,
        },
        /// Formosa dataset options.
        Iii {
            classes_file: PathBuf,
            image_size: NonZeroUsize,
            dataset_dir: PathBuf,
            #[serde(default = "empty_hashset::<PathBuf>")]
            blacklist_files: HashSet<PathBuf>,
        },
        /// CSV dataset options.
        Csv {
            image_dir: PathBuf,
            label_file: PathBuf,
            classes_file: PathBuf,
            image_size: NonZeroUsize,
            input_channels: NonZeroUsize,
        },
    }
}

mod preprocess {
    use super::*;

    /// Input data preprocessing options.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PreprocessConfig {
        // /// The diretory to save the data cache. SSD-backed filesystem and tmpfs are suggested.
        // pub cache_dir: PathBuf,
        /// The factor that tolerates out-of-image boundary bounding boxes.
        pub out_of_bound_tolerance: R64,
        /// The minimum bounding box size in ratio unit.
        pub min_bbox_size: Ratio,
        /// The minimum ratio of preserving area after a bounding box is cropped.
        pub min_bbox_cropping_ratio: Ratio,
        /// The device where the preprocessor works on.
        #[serde(with = "tch_serde::serde_device")]
        pub device: Device,
    }
}

mod output {
    use super::*;

    /// Model configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct OutputConfig {
        pub output_dir: PathBuf,
        pub nms_iou_thresh: R64,
        pub nms_conf_thresh: R64,
    }
}

fn empty_hashset<T>() -> HashSet<T> {
    HashSet::new()
}

pub fn deserialize_version<'de, D>(deserializer: D) -> Result<Version, D::Error>
where
    D: Deserializer<'de>,
{
    let text = String::deserialize(deserializer)?;
    let version = Version::parse(&text).map_err(|err| {
        D::Error::custom(format!(
            "failed to parse version number '{}': {:?}",
            text, err
        ))
    })?;

    if !CONFIG_VERSION.matches(&version) {
        return Err(D::Error::custom(format!(
            "incompatible version: get '{}', but it is incompatible with requirement '{}'",
            version, &*CONFIG_VERSION,
        )));
    }

    Ok(version)
}

mod serde_vec_device {
    use super::*;

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    struct DeviceWrapper(#[serde(with = "tch_serde::serde_device")] Device);

    pub fn serialize<S>(devices: &Vec<Device>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let devices: Vec<_> = devices.iter().cloned().map(DeviceWrapper).collect();
        devices.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Device>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let devices = Vec::<DeviceWrapper>::deserialize(deserializer)?;
        let devices: Vec<_> = devices
            .into_iter()
            .map(|DeviceWrapper(device)| device)
            .collect();
        Ok(devices)
    }
}
