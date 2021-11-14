pub use anyhow::{bail, ensure, format_err, Context, Error, Result};
pub use approx::{abs_diff_eq, assert_abs_diff_eq, AbsDiffEq};
pub use argh::FromArgs;
pub use async_std::sync::RwLock;
pub use chrono::{DateTime, Local};
pub use coco::Category;
pub use cv_convert::TryIntoCv;
pub use dashmap::DashSet;
pub use derivative::Derivative;
pub use futures::{
    future,
    future::FutureExt,
    stream::{self, Stream, StreamExt, TryStreamExt},
    AsyncWriteExt,
};
pub use getset::{CopyGetters, Getters};
pub use indexmap::{IndexMap, IndexSet};
pub use itertools::{chain, iproduct, izip, Itertools};
pub use lazy_static::lazy_static;
pub use log::{error, info, warn};
pub use ndarray::{Array, Array2, Array3, Array5, ArrayD};
pub use noisy_float::prelude::*;
pub use owning_ref::ArcRef;
pub use par_stream::prelude::*;
pub use percent_encoding::NON_ALPHANUMERIC;
pub use petgraph::graphmap::DiGraphMap;
pub use rand::{prelude::*, rngs::StdRng};
pub use regex::Regex;
pub use rusty_perm::{prelude::*, PermD};
pub use serde::{
    de::{Error as _, Visitor},
    ser::{Error as _, SerializeSeq},
    Deserialize, Deserializer, Serialize, Serializer,
};
pub use slice_of_array::{SliceFlatExt, SliceNestExt};
pub use std::{
    borrow::{Borrow, Cow},
    cmp::{self, Ordering},
    collections::{hash_map, HashMap, HashSet},
    convert::{TryFrom, TryInto},
    env::{self, VarError},
    fmt,
    fmt::Debug,
    future::Future,
    hash::{Hash, Hasher},
    iter::{self, FromIterator, Sum},
    marker::PhantomData,
    mem,
    num::NonZeroUsize,
    ops::{Add, Deref, DerefMut, Div, Index, Mul, Range, Rem, Sub},
    path::{Path, PathBuf},
    pin::Pin,
    rc::Rc,
    sync::{
        atomic::{self, AtomicBool},
        Arc, Once,
    },
    time::{Duration, Instant},
};
pub use strum::AsRefStr;
pub use tch::{
    kind::{FLOAT_CPU, INT64_CPU},
    nn::{self, OptimizerConfig},
    vision, Device, IndexOp, Kind, Reduction, Tensor,
};
pub use tch_goodies::{
    Activation, CyCxHW, CyCxHWTensor, CyCxHWTensorUnchecked, GridCyCxHW, GridSize, LabelTensor,
    ObjectDetectionTensor, ObjectDetectionTensorUnchecked, PixelCyCxHW, PixelRectLabel,
    PixelRectTransform, PixelSize, PixelTLBR, Ratio, RatioCyCxHW, RatioRectLabel, RatioSize,
    Rect as _, TLBRTensor, TLBRTensorUnchecked, TensorExt, UnitlessCyCxHW, UnitlessTLBR, TLBR,
};
pub use tch_tensor_like::TensorLike;
pub use tfrecord::EventWriterInit;
pub use tokio::sync::{broadcast, mpsc};
pub use tracing::instrument;
pub use uuid::Uuid;

pub type Fallible<T> = Result<T, Error>;

unzip_n::unzip_n!(pub 2);
unzip_n::unzip_n!(pub 3);
unzip_n::unzip_n!(pub 4);
unzip_n::unzip_n!(pub 5);
unzip_n::unzip_n!(pub 6);
unzip_n::unzip_n!(pub 7);
unzip_n::unzip_n!(pub 8);
unzip_n::unzip_n!(pub 9);
unzip_n::unzip_n!(pub 10);
unzip_n::unzip_n!(pub 11);
