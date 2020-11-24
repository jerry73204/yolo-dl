pub use anyhow::{bail, ensure, format_err, Context, Error, Result};
pub use argh::FromArgs;
pub use arraystring::ArrayString;
pub use futures::{
    stream, AsyncReadExt, AsyncSeekExt, AsyncWriteExt, StreamExt, TryStream, TryStreamExt,
};
pub use indexmap::{IndexMap, IndexSet};
pub use indicatif::{ProgressBar, ProgressStyle};
pub use itertools::Itertools;
pub use log::{info, warn};
pub use memmap::{Mmap, MmapOptions};
pub use par_stream::{ParStreamExt, TryParStreamExt};
pub use prettytable::{cell, row, Table};
pub use safe_transmute::TriviallyTransmutable;
pub use serde::{
    de::Error as _, ser::Error as _, Deserialize, Deserializer, Serialize, Serializer,
};
pub use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet},
    convert::TryFrom,
    error,
    io::prelude::*,
    iter,
    marker::{PhantomData, Unpin},
    mem,
    ops::Range,
    path::{Path, PathBuf},
    slice,
    sync::Arc,
};
pub use tch::{vision, Device, Kind};
pub use tch_goodies::TensorExt;
pub use voc_dataset as voc;

pub type Fallible<T> = Result<T, Error>;
