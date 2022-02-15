use crate::common::*;

pub trait Element: Float {}

impl<T> Element for T where T: Float {}
