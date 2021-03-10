//! Common imports from external crates.

pub use anyhow::{bail, ensure, format_err, Context, Error, Result};
pub use approx::{abs_diff_eq, AbsDiffEq};
pub use chrono::{DateTime, Local};
pub use futures::{
    future,
    future::FutureExt,
    stream::{self, Stream, StreamExt, TryStreamExt},
    AsyncWriteExt,
};
pub use image::{imageops::FilterType, DynamicImage, FlatSamples, ImageFormat, Pixel};
pub use indexmap::IndexSet;
pub use itertools::{izip, Itertools};
pub use ndarray::{Array, Array2, Array3, ArrayD};
pub use noisy_float::prelude::*;
pub use once_cell::sync::Lazy;
pub use owning_ref::ArcRef;
pub use par_stream::{ParStreamConfig, ParStreamExt, TryParStreamExt};
pub use percent_encoding::NON_ALPHANUMERIC;
pub use rand::{distributions::Distribution, prelude::*, rngs::StdRng, seq::SliceRandom};
pub use regex::Regex;
pub use semver::{Version, VersionReq};
pub use serde::{
    de::{Error as DeserializeError, Visitor},
    ser::{Error as SerializeError, SerializeSeq},
    Deserialize, Deserializer, Serialize, Serializer,
};
pub use std::{
    borrow::{Borrow, Cow},
    cmp::Ordering,
    collections::{HashMap, HashSet},
    convert::{TryFrom, TryInto},
    env::{self, VarError},
    fmt::Debug,
    fs,
    future::Future,
    hash::{Hash, Hasher},
    iter::{self, Sum},
    marker::PhantomData,
    mem,
    num::NonZeroUsize,
    ops::{Add, Deref, DerefMut, Div, Mul, Rem, Sub},
    path::{Path, PathBuf},
    pin::Pin,
    rc::Rc,
    sync::{
        atomic::{self, AtomicBool},
        Arc, Once,
    },
    time::{Duration, Instant},
};
pub use structopt::StructOpt;
pub use tch::{
    kind::FLOAT_CPU,
    nn::{self, OptimizerConfig as _},
    vision, Device, IndexOp, Kind, Reduction, Tensor,
};
pub use tch_goodies::{
    CyCxHW, GridCyCxHW, GridLabel, GridSize, PixelCyCxHW, PixelLabel, PixelSize, PixelTLBR, Ratio,
    RatioCyCxHW, RatioLabel, TLBRTensor, TensorExt, NONE_INDEX, TLBR,
};
pub use tch_tensor_like::TensorLike;
pub use tfrecord::{EventWriter, EventWriterInit};
pub use tokio::sync::{broadcast, mpsc};
pub use tracing::{error, info, info_span, instrument, trace, trace_span, warn, Instrument};
pub use tracing_subscriber::{prelude::*, EnvFilter};
pub use uuid::Uuid;
pub use yolo_dl::{
    dataset::{
        CachedDataset, CocoDataset, CsvDataset, DataRecord, GenericDataset, IiiDataset,
        RandomAccessDataset, SanitizedDataset, VocDataset,
    },
    loss::{
        MatchingOutput, YoloBenchmark, YoloBenchmarkInit, YoloBenchmarkOutput, YoloInference,
        YoloInferenceInit, YoloInferenceOutput, YoloLoss, YoloLossAuxiliary, YoloLossInit,
        YoloLossOutput,
    },
    model::{DetectionInfo, FlatIndex, InstanceIndex, MergeDetect2DOutput, YoloModel},
    processor::{CacheLoader, ColorJitterInit, ParallelMosaicProcessorInit, RandomAffineInit},
    profiling::Timing,
};

pub type Fallible<T> = Result<T, Error>;

unzip_n::unzip_n!(pub 2);
unzip_n::unzip_n!(pub 3);
unzip_n::unzip_n!(pub 4);
unzip_n::unzip_n!(pub 5);
unzip_n::unzip_n!(pub 6);
unzip_n::unzip_n!(pub 7);
