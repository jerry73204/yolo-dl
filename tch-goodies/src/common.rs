pub use anyhow::{bail, ensure, format_err, Context, Error, Result};
pub use approx::{abs_diff_eq, AbsDiffEq};
pub use cv_convert::{IntoCv, TryIntoCv};
pub use getset::{CopyGetters, Getters};
pub use image::{DynamicImage, GenericImageView, ImageBuffer, Pixel};
pub use itertools::{iproduct, izip, Itertools};
pub use log::warn;
pub use maplit::hashset;
pub use mona::prelude::*;
pub use ndarray::Array5;
pub use noisy_float::prelude::*;
pub use num_traits::{FromPrimitive, Num, NumCast, NumOps, One, ToPrimitive, Zero};
pub use serde::{
    de::Error as _, ser::Error as _, Deserialize, Deserializer, Serialize, Serializer,
};
pub use std::{
    borrow::Borrow,
    cmp::Ordering,
    collections::HashSet,
    convert::{TryFrom, TryInto},
    fmt::{self, Display, Formatter},
    iter::{self, FromIterator},
    marker::PhantomData,
    num::FpCategory,
    ops::{Add, Deref, Div, Mul, Neg, Range, RangeInclusive, Rem, Sub},
    sync::Once,
};
pub use tch::{
    kind::Element,
    nn::{self, Module, ModuleT},
    vision, Device, IndexOp, Kind, Tensor,
};
pub use tch_tensor_like::TensorLike;

pub type Fallible<T> = Result<T, Error>;

unzip_n::unzip_n!(pub 2);
unzip_n::unzip_n!(pub 3);
unzip_n::unzip_n!(pub 4);
unzip_n::unzip_n!(pub 5);
unzip_n::unzip_n!(pub 7);
unzip_n::unzip_n!(pub 9);
unzip_n::unzip_n!(pub 10);
unzip_n::unzip_n!(pub 11);
