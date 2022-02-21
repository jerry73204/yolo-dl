use crate::common::*;
use bbox::CyCxHW;
use label::Label;
use tch_goodies::{Pixel, Ratio};

pub type RatioLabel = Ratio<Label<CyCxHW<R64>, usize>>;
pub type PixelLabel = Pixel<Label<CyCxHW<R64>, usize>>;
