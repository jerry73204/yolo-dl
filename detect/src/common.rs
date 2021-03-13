pub use anyhow::{format_err, Error, Result};
pub use futures::{
    future::FutureExt,
    stream::{self, Stream, StreamExt, TryStreamExt},
};
pub use itertools::Itertools;
pub use noisy_float::prelude::*;
pub use once_cell::sync::Lazy;
pub use par_stream::{ParStreamExt, TryParStreamExt};
pub use semver::{Version, VersionReq};
pub use serde::{
    de::{Error as DeserializeError, Visitor},
    ser::{Error as SerializeError, SerializeSeq},
    Deserialize, Deserializer, Serialize, Serializer,
};
pub use std::{
    collections::HashSet,
    convert::TryInto,
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
};
pub use tch::{nn, Device, IndexOp, Tensor};
pub use tch_goodies::{Ratio, RatioCyCxHW, RatioLabel};
pub use tch_tensor_like::TensorLike;
pub use yolo_dl::{
    dataset::{
        CocoDataset, CsvDataset, DataRecord, IiiDataset, OnDemandDataset, RandomAccessStream,
        SanitizedDataset, StreamingDataset, VocDataset,
    },
    loss::YoloInferenceInit,
    model::YoloModel,
};

pub type Fallible<T> = Result<T, Error>;

unzip_n::unzip_n!(pub 3);
