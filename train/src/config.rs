use crate::common::*;
use yolo_dl::loss::{IoUKind, MatchGrid};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub dir: PathBuf,
    pub enable_images: bool,
    pub enable_training_output: bool,
    pub enable_debug_stat: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub class_whitelist: Option<HashSet<String>>,
    pub kind: DatasetKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DatasetKind {
    Coco {
        classes_file: PathBuf,
        image_size: NonZeroUsize,
        dataset_dir: PathBuf,
        dataset_name: String,
    },
    Voc {
        classes_file: PathBuf,
        image_size: NonZeroUsize,
        dataset_dir: PathBuf,
    },
    Iii {
        classes_file: PathBuf,
        image_size: NonZeroUsize,
        dataset_dir: PathBuf,
        #[serde(default = "empty_hashset::<PathBuf>")]
        blacklist_files: HashSet<PathBuf>,
    },
    Mmap {
        dataset_file: PathBuf,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessorConfig {
    pub cache_dir: PathBuf,
    pub mixup_prob: Ratio,
    pub cutmix_prob: Ratio,
    pub mosaic_prob: Ratio,
    pub mosaic_margin: Ratio,
    // pub hsv_distort_prob: Ratio,
    // pub max_hue_shift: R64,
    // pub max_saturation_scale: R64,
    // pub max_value_scale: R64,
    pub affine_prob: Ratio,
    pub rotate_prob: Option<Ratio>,
    pub rotate_degrees: Option<R64>,
    pub translation_prob: Option<Ratio>,
    pub translation: Option<R64>,
    pub scale_prob: Option<Ratio>,
    pub scale: Option<(R64, R64)>,
    // pub shear_prob: Option<Ratio>,
    // pub shear: Option<R64>,
    pub horizontal_flip_prob: Option<Ratio>,
    pub vertical_flip_prob: Option<Ratio>,
    pub bbox_scaling: R64,
    pub out_of_bound_tolerance: R64,
    pub min_bbox_size: Ratio,
    #[serde(with = "tch_serde::serde_device")]
    pub device: Device,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: NonZeroUsize,
    pub default_minibatch_size: NonZeroUsize,
    #[serde(default = "default_initial_step")]
    pub initial_step: usize,
    pub lr_schedule: LearningRateSchedule,
    pub momentum: R64,
    pub weight_decay: R64,
    pub loss: LossConfig,
    pub save_checkpoint_steps: Option<NonZeroUsize>,
    pub load_checkpoint: LoadCheckpoint,
    pub enable_multi_gpu: bool,
    #[serde(with = "tch_serde::serde_device")]
    pub master_device: Device,
    pub workers: Vec<WorkerConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossConfig {
    pub match_grid_method: MatchGrid,
    pub iou_kind: IoUKind,
    pub iou_loss_weight: Option<R64>,
    pub objectness_loss_weight: Option<R64>,
    pub classification_loss_weight: Option<R64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LoadCheckpoint {
    Disabled,
    FromRecent,
    FromFile { file: PathBuf },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    #[serde(with = "tch_serde::serde_device")]
    pub device: Device,
    pub minibatch_size: Option<NonZeroUsize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LearningRateSchedule {
    Constant { lr: R64 },
    StepWise { steps: Vec<(usize, R64)> },
}

fn empty_hashset<T>() -> HashSet<T> {
    HashSet::new()
}

fn default_initial_step() -> usize {
    info!("use default initail step 0");
    0
}
