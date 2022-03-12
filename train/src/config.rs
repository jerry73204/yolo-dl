//! Training program configuration format.

use crate::common::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_semver::SemverReq;
use tch_goodies::lr_schedule;
use yolo_dl::{loss, processor::ColorJitterInit};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, SemverReq)]
#[version("0.1.0")]
pub struct ConfigVersion;

/// The main training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub version: ConfigVersion,
    pub model: Model,
    pub dataset: Dataset,
    pub logging: Logging,
    pub preprocessor: Preprocessor,
    pub training: Training,
    pub benchmark: Benchmark,
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

pub use model::*;
mod model {
    use super::*;

    /// The model configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "kind")]
    pub enum Model {
        Darknet(DarknetModel),
        NewslabV1(NewslabV1Model),
    }

    /// The Darknet variant model configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DarknetModel {
        pub cfg_file: PathBuf,
    }

    /// The NEWSLAB variant model configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct NewslabV1Model {
        pub cfg_file: PathBuf,
    }
}

pub use dataset::*;
mod dataset {
    use super::*;

    /// Dataset options.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Dataset {
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

pub use preprocessor::*;
mod preprocessor {
    use super::*;

    /// Data preprocessing options.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Preprocessor {
        pub pipeline: Pipeline,
        pub cache: Cache,
        pub mixup: MixUp,
        pub random_affine: RandomAffine,
        pub color_jitter: ColorJitter,
        pub cleanse: Cleanse,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Pipeline {
        /// If set, process image records without ordering.
        pub unordered_records: bool,
        /// If set, produce training batches without ordering.
        pub unordered_batches: bool,
        /// The maximum number of waiting data records per preprocessing stage.
        pub worker_buf_size: Option<usize>,
        /// The device where the preprocessor works on.
        #[serde(with = "tch_serde::serde_device")]
        pub device: Device,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "method")]
    pub enum Cache {
        NoCache,
        FileCache {
            /// The diretory to save the data cache. SSD-backed filesystem and tmpfs are suggested.
            cache_dir: PathBuf,
        },
        MemoryCache,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MixUp {
        /// The probability to apply Mix-Up.
        pub mixup_prob: R64,
        /// The probability to apply Cut-Mix.
        pub cutmix_prob: R64,
        /// The probability to apply Mosaic mixing.
        pub mosaic_prob: R64,
        /// The minimum offset from image boundary of pivot point in Mosaic mixing.
        ///
        /// It is specified in ratio unit.
        pub mosaic_margin: R64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RandomAffine {
        /// The probability to apply random affine transformation.
        pub affine_prob: R64,
        /// The probability to apply random rotation.
        pub rotate_prob: Option<R64>,
        /// The maximum degrees of random rotation.
        pub rotate_degrees: Option<R64>,
        /// The probability to apply random translation.
        pub translation_prob: Option<R64>,
        /// The maximum distance of random translation in ratio unit.
        pub translation: Option<R64>,
        /// The probability to apply random scaling.
        pub scale_prob: Option<R64>,
        /// The pair of minimum and maximum scaling ratio.
        pub scale: Option<(R64, R64)>,
        // pub shear_prob: Option<R64>,
        // pub shear: Option<R64>,
        /// The probability to apply horizontal flip.
        pub horizontal_flip_prob: Option<R64>,
        /// The probability to apply vertical flip.
        pub vertical_flip_prob: Option<R64>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ColorJitter {
        pub color_jitter_prob: R64,
        pub hue_shift: Option<R64>,
        pub saturation_shift: Option<R64>,
        pub value_shift: Option<R64>,
    }

    impl ColorJitter {
        pub fn color_jitter_init(&self) -> ColorJitterInit {
            let Self {
                hue_shift,
                saturation_shift,
                value_shift,
                ..
            } = *self;

            ColorJitterInit {
                hue_shift,
                saturation_shift,
                value_shift,
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Cleanse {
        /// The scaling factor of bounding box size.
        pub bbox_scaling: R64,
        /// The factor that tolerates out-of-image boundary bounding boxes.
        pub out_of_bound_tolerance: R64,
        /// The minimum bounding box size in ratio unit.
        pub min_bbox_size: R64,
        /// The minimum ratio of preserving area after a bounding box is cropped.
        pub min_bbox_cropping_ratio: R64,
    }
}

pub use training::*;
mod training {

    use super::*;

    /// The training options.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Training {
        /// The batch size.
        pub batch_size: NonZeroUsize,
        /// If enabled, it overrides the initial training step.
        pub override_initial_step: Option<usize>,
        /// If set, it saves a checkpoint file per this steps.
        pub save_checkpoint_steps: Option<NonZeroUsize>,
        /// Checkpoint file loading method.
        pub load_checkpoint: LoadCheckpoint,
        /// Training device options.
        pub device_config: DeviceConfig,
        pub optimizer: Optimizer,
        /// The loss function options.
        pub loss: loss::Config,
    }

    /// Training device options.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum DeviceConfig {
        /// Use single device.
        SingleDevice {
            #[serde(with = "tch_serde::serde_device")]
            device: Device,
        },
        /// Use multiple device with uniform mini-batch size on each device.
        MultiDevice {
            #[serde(with = "serde_vec_device")]
            devices: Vec<Device>,
        },
        /// Use multiple device with mini-batch size set for each device.
        NonUniformMultiDevice { devices: Vec<Worker> },
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Worker {
        #[serde(with = "tch_serde::serde_device")]
        pub device: Device,
        pub minibatch_size: NonZeroUsize,
    }

    /// Checkpoint file loading method.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum LoadCheckpoint {
        /// Disable checkpoint file loading.
        Disabled,
        /// Load the most recent checkpoint file.
        FromRecent,
        /// Load the checkpoint file at specified path.
        FromFile { file: PathBuf },
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Optimizer {
        /// Learning rate scheduling strategy.
        pub lr_schedule: lr_schedule::Config,
        /// The momentum parameter for optimizer.
        pub momentum: R64,
        /// The weight decay parameter for optimizer.
        pub weight_decay: R64,
        /// The maximum of absolute gradient values.
        pub clip_grad: Option<R64>,
    }
}

/// Data logging options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Logging {
    pub dir: PathBuf,
    pub enable_images: bool,
    pub enable_debug_stat: bool,
    pub enable_inference: bool,
    pub enable_benchmark: bool,
    pub enable_gradients: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Benchmark {
    pub nms_iou_thresh: R64,
    pub nms_conf_thresh: R64,
}

fn empty_hashset<T>() -> HashSet<T> {
    HashSet::new()
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
