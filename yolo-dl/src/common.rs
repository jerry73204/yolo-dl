pub use anyhow::{bail, ensure, format_err, Context as _, Error, Result};
pub use approx::{abs_diff_eq, assert_abs_diff_eq};
pub use futures::{
    future,
    future::FutureExt as _,
    stream::{self, Stream, StreamExt as _, TryStreamExt as _},
    AsyncWriteExt as _,
};
pub use indexmap::{IndexMap, IndexSet};
pub use itertools::{chain, iproduct, izip, Itertools as _};
pub use lazy_static::lazy_static;
pub use log::{error, info, warn};
pub use noisy_float::prelude::*;
pub use owning_ref::ArcRef;
pub use par_stream::prelude::*;
pub use rand::prelude::*;
pub use serde::{
    de::{Error as _, Visitor},
    ser::{Error as _, SerializeSeq},
    Deserialize, Deserializer, Serialize, Serializer,
};
pub use slice_of_array::{SliceFlatExt as _, SliceNestExt as _};
pub use std::{
    borrow::{Borrow, Cow},
    cmp::{self, Ordering},
    collections::{hash_map, HashMap, HashSet},
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
    str::FromStr,
    sync::{
        atomic::{self, AtomicBool},
        Arc, Once,
    },
    time::{Duration, Instant},
};
pub use tch::{
    kind::{FLOAT_CPU, INT64_CPU},
    nn::{self, OptimizerConfig},
    vision, Device, IndexOp, Kind, Reduction, Tensor,
};
pub use tch_tensor_like::TensorLike;

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
