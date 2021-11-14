pub use anyhow::{format_err, Error, Result};
pub use cv_convert::{IntoCv, ShapeConvention, TensorAsImage, TryIntoCv};
pub use futures::{
    future::FutureExt,
    stream::{self, Stream, StreamExt, TryStreamExt},
};
pub use itertools::Itertools;
pub use noisy_float::prelude::*;
pub use once_cell::sync::Lazy;
pub use opencv::{
    core::{Mat, Rect, Scalar, Vector},
    imgcodecs, imgproc,
};
pub use par_stream::prelude::*;
pub use semver::{Version, VersionReq};
pub use serde::{
    de::{Error as DeserializeError, Visitor},
    ser::{Error as SerializeError, SerializeSeq},
    Deserialize, Deserializer, Serialize, Serializer,
};
pub use std::{
    collections::HashSet,
    convert::{TryFrom, TryInto},
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
};
pub use tch::{nn, Device, IndexOp, Tensor};
pub use tch_goodies::{
    PixelCyCxHW, PixelRectLabel, PixelSize, Ratio, RatioCyCxHW, RatioRectLabel, Rect as _,
};
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
