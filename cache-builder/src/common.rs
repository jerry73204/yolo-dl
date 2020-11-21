pub use anyhow::{bail, ensure, Context, Error, Result};
pub use argh::FromArgs;
pub use arraystring::ArrayString;
pub use futures::{io::Cursor, stream, AsyncWriteExt, StreamExt, TryStream, TryStreamExt};
pub use indexmap::IndexMap;
pub use itertools::Itertools;
pub use log::{info, warn};
pub use memmap::MmapOptions;
pub use par_stream::{ParStreamExt, TryParStreamExt};
pub use safe_transmute::TriviallyTransmutable;
pub use serde::{Deserialize, Serialize};
pub use std::{
    collections::{HashMap, HashSet},
    error,
    marker::{PhantomData, Unpin},
    mem,
    path::{Path, PathBuf},
    slice,
    sync::Arc,
};
pub use voc_dataset as voc;

pub type Fallible<T> = Result<T, Error>;
