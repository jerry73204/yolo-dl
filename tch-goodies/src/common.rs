pub use anyhow::{bail, ensure, format_err, Context, Error, Result};
pub use approx::{abs_diff_eq, AbsDiffEq};
pub use getset::Getters;
pub use image::{DynamicImage, GenericImageView, ImageBuffer, Pixel};
pub use itertools::{izip, Itertools};
pub use maplit::hashset;
pub use noisy_float::prelude::*;
pub use num_traits::{FromPrimitive, Num, NumOps, One, Zero};
pub use serde::{
    de::Error as _, ser::Error as _, Deserialize, Deserializer, Serialize, Serializer,
};
pub use std::{
    borrow::Borrow,
    cmp::Ordering,
    collections::HashSet,
    convert::{TryFrom, TryInto},
    fmt::{self, Display, Formatter},
    iter,
    marker::PhantomData,
    ops::{Add, Deref, Div, Mul, Range, RangeInclusive, Rem, Sub},
};
pub use tch::{kind::Element, vision, Device, IndexOp, Kind, Tensor};
pub use tch_tensor_like::TensorLike;

pub type Fallible<T> = Result<T, Error>;

unzip_n::unzip_n!(pub 2);
unzip_n::unzip_n!(pub 4);
unzip_n::unzip_n!(pub 10);
