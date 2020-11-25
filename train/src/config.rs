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
    pub save_images: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub classes_file: PathBuf,
    pub image_size: NonZeroUsize,
    pub class_whitelist: Option<HashSet<String>>,
    pub kind: DatasetKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum DatasetKind {
    Coco {
        dataset_dir: PathBuf,
        dataset_name: String,
    },
    Voc {
        dataset_dir: PathBuf,
    },
    Iii {
        dataset_dir: PathBuf,
        #[serde(default = "empty_hashset::<PathBuf>")]
        blacklist_files: HashSet<PathBuf>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessorConfig {
    pub cache_dir: PathBuf,
    pub mosaic_prob: Ratio,
    pub mosaic_margin: Ratio,
    pub affine_prob: Ratio,
    pub rotate_degrees: Option<R64>,
    pub translation: Option<R64>,
    pub scale: Option<(R64, R64)>,
    pub shear: Option<R64>,
    pub horizontal_flip: bool,
    pub vertical_flip: bool,
    pub bbox_scaling: R64,
    #[serde(with = "tch_serde::serde_device")]
    pub device: Device,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: NonZeroUsize,
    pub default_minibatch_size: NonZeroUsize,
    pub match_grid_method: MatchGrid,
    pub iou_kind: IoUKind,
    pub initial_step: Option<usize>,
    pub lr_schedule: LearningRateSchedule,
    pub momentum: R64,
    pub weight_decay: R64,
    pub save_checkpoint_steps: Option<NonZeroUsize>,
    pub load_checkpoint: LoadCheckpoint,
    pub enable_multi_gpu: bool,
    #[serde(with = "tch_serde::serde_device")]
    pub master_device: Device,
    pub workers: Vec<WorkerConfig>,
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
