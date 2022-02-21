//! Common imports from external crates.

pub use anyhow::{bail, ensure, format_err, Context as _, Error, Result};
pub use approx::{abs_diff_eq, AbsDiffEq as _};
pub use chrono::{DateTime, Local};
pub use collected::MinVal;
pub use derivative::Derivative;
pub use futures::{
    future::{self, FutureExt as _},
    stream::{self, BoxStream, Stream, StreamExt as _, TryStreamExt as _},
    AsyncWriteExt as _,
};
pub use indexmap::IndexSet;
pub use itertools::{izip, Itertools as _};
pub use log::{error, info, trace, warn};
pub use noisy_float::prelude::*;
pub use once_cell::sync::Lazy;
pub use owning_ref::ArcRef;
pub use par_stream::prelude::*;
pub use rand::{
    distributions::Distribution,
    prelude::*,
    rngs::{OsRng, StdRng},
    seq::SliceRandom as _,
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
pub use tch_tensor_like::TensorLike;
pub use tokio::sync::{broadcast, mpsc};

use unzip_n::unzip_n;
unzip_n!(pub 2);
unzip_n!(pub 3);
unzip_n!(pub 4);
unzip_n!(pub 5);
unzip_n!(pub 6);
unzip_n!(pub 7);
