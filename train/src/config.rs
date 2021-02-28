//! Training program configuration format.

use crate::common::*;
use yolo_dl::loss::{BoxMetric, ClassificationLossKind, MatchGrid, ObjectnessLossKind};

pub use dataset::*;
pub use model::*;
pub use training::*;

/// The main training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model: ModelConfig,
    pub dataset: DatasetConfig,
    pub logging: LoggingConfig,
    pub preprocessor: PreprocessorConfig,
    pub training: TrainingConfig,
}

impl Config {
    pub fn open<P>(path: P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let text = std::fs::read_to_string(path)?;
        let config = json5::from_str(&text)?;
        Ok(config)
    }
}

mod model {
    use super::*;

    /// The model configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "kind")]
    pub enum ModelConfig {
        Darknet(DarknetModelConfig),
        NewslabV1(NewslabV1ModelConfig),
    }

    /// The Darknet variant model configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DarknetModelConfig {
        pub cfg_file: PathBuf,
    }

    /// The NEWSLAB variant model configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct NewslabV1ModelConfig {
        pub cfg_file: PathBuf,
    }
}

/// Data logging options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub dir: PathBuf,
    pub enable_images: bool,
    pub enable_debug_stat: bool,
}

mod dataset {
    use super::*;

    /// Dataset options.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DatasetConfig {
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

/// Data preprocessing options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessorConfig {
    /// If set, process image records without ordering.
    pub unordered_records: bool,
    /// If set, produce training batches without ordering.
    pub unordered_batches: bool,
    /// The maximum number of waiting data records per preprocessing stage.
    pub worker_buf_size: Option<usize>,
    /// The diretory to save the data cache. SSD-backed filesystem and tmpfs are suggested.
    pub cache_dir: PathBuf,
    pub color_jitter_prob: Ratio,
    pub hue_shift: Option<R64>,
    pub saturation_shift: Option<R64>,
    pub value_shift: Option<R64>,
    /// The probability to apply Mix-Up.
    pub mixup_prob: Ratio,
    /// The probability to apply Cut-Mix.
    pub cutmix_prob: Ratio,
    /// The probability to apply Mosaic mixing.
    pub mosaic_prob: Ratio,
    /// The minimum offset from image boundary of pivot point in Mosaic mixing.
    ///
    /// It is specified in ratio unit.
    pub mosaic_margin: Ratio,
    /// The probability to apply random affine transformation.
    pub affine_prob: Ratio,
    /// The probability to apply random rotation.
    pub rotate_prob: Option<Ratio>,
    /// The maximum degrees of random rotation.
    pub rotate_degrees: Option<R64>,
    /// The probability to apply random translation.
    pub translation_prob: Option<Ratio>,
    /// The maximum distance of random translation in ratio unit.
    pub translation: Option<R64>,
    /// The probability to apply random scaling.
    pub scale_prob: Option<Ratio>,
    /// The pair of minimum and maximum scaling ratio.
    pub scale: Option<(R64, R64)>,
    // pub shear_prob: Option<Ratio>,
    // pub shear: Option<R64>,
    /// The probability to apply horizontal flip.
    pub horizontal_flip_prob: Option<Ratio>,
    /// The probability to apply vertical flip.
    pub vertical_flip_prob: Option<Ratio>,
    /// The scaling factor of bounding box size.
    pub bbox_scaling: R64,
    /// The factor that tolerates out-of-image boundary bounding boxes.
    pub out_of_bound_tolerance: R64,
    /// The minimum bounding box size in ratio unit.
    pub min_bbox_size: Ratio,
    /// The device where the preprocessor works on.
    #[serde(with = "tch_serde::serde_device")]
    pub device: Device,
}

mod training {
    use super::*;

    /// The training options.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TrainingConfig {
        /// The batch size.
        pub batch_size: NonZeroUsize,
        /// If enabled, it overrides the initial training step.
        pub override_initial_step: Option<usize>,
        /// Learning rate scheduling strategy.
        pub lr_schedule: LearningRateSchedule,
        /// The momentum parameter for optimizer.
        pub momentum: R64,
        /// The weight decay parameter for optimizer.
        pub weight_decay: R64,
        /// The loss function options.
        pub loss: LossConfig,
        /// If set, it saves a checkpoint file per this steps.
        pub save_checkpoint_steps: Option<NonZeroUsize>,
        /// Checkpoint file loading method.
        pub load_checkpoint: LoadCheckpoint,
        /// Training device options.
        pub device_config: DeviceConfig,
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
            minibatch_size: NonZeroUsize,
            #[serde(with = "serde_vec_device")]
            devices: Vec<Device>,
        },
        /// Use multiple device with mini-batch size set for each device.
        NonUniformMultiDevice { devices: Vec<WorkerConfig> },
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WorkerConfig {
        #[serde(with = "tch_serde::serde_device")]
        pub device: Device,
        pub minibatch_size: NonZeroUsize,
    }

    /// The learning rate scheduling strategy.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum LearningRateSchedule {
        /// Use constant learning rate.
        Constant { lr: R64 },
        /// Use specific learning rate at specified steps.
        StepWise { steps: Vec<(usize, R64)> },
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

    /// The loss function configuration.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LossConfig {
        pub objectness_loss_kind: ObjectnessLossKind,
        pub classification_loss_kind: ClassificationLossKind,
        /// The method to match ground truth to predicted bounding boxes.
        pub match_grid_method: MatchGrid,
        /// The choice of bounding box metric.
        pub box_metric: BoxMetric,
        /// The weighting factor of IoU loss.
        pub iou_loss_weight: Option<R64>,
        /// The weighting factor of objectness loss.
        pub objectness_loss_weight: Option<R64>,
        /// The weighting factor of classification loss.
        pub classification_loss_weight: Option<R64>,
    }
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
