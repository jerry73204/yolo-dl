pub use anyhow::{bail, ensure, format_err, Context, Error, Result};
pub use argh::FromArgs;
pub use arraystring::ArrayString;
pub use futures::{io::Cursor, stream, AsyncWriteExt, StreamExt, TryStream, TryStreamExt};
pub use indexmap::{IndexMap, IndexSet};
pub use itertools::Itertools;
pub use log::{info, warn};
pub use memmap::MmapOptions;
pub use par_stream::{ParStreamExt, TryParStreamExt};
pub use safe_transmute::TriviallyTransmutable;
pub use serde::{Deserialize, Serialize};
pub use std::{
    collections::{HashMap, HashSet},
    convert::TryFrom,
    error,
    marker::{PhantomData, Unpin},
    mem,
    path::{Path, PathBuf},
    slice,
    sync::Arc,
};
pub use tch::{vision, Kind};
pub use tch_goodies::TensorExt;
pub use voc_dataset as voc;

pub type Fallible<T> = Result<T, Error>;
