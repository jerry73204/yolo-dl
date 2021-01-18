use crate::common::*;

pub trait AsXY<TX, TY> {
    fn x(&self) -> TX;
    fn y(&self) -> TY;
}

impl<T, TX, TY> AsXY<TX, TY> for &T
where
    T: AsXY<TX, TY>,
{
    fn x(&self) -> TX {
        (*self).x()
    }
    fn y(&self) -> TY {
        (*self).y()
    }
}

impl<TX, TY> AsXY<TX, TY> for (TX, TY)
where
    TX: Copy,
    TY: Copy,
{
    fn x(&self) -> TX {
        self.0
    }

    fn y(&self) -> TY {
        self.1
    }
}

pub fn interpolate_slice<T>(points: impl IntoIterator<Item = R64>, values: &[T]) -> Vec<(R64, R64)>
where
    T: AsXY<R64, R64>,
{
    let mut points_iter = points.into_iter();
    let mut values_iter = values.iter().zip(values.iter().skip(1));

    let (mut former, mut latter) = values_iter.next().unwrap();
    let mut inter_x = points_iter.next().unwrap();

    let mut interpolated = vec![];

    loop {
        match (former.x() <= inter_x, inter_x <= latter.x()) {
            (false, false) => match points_iter.next() {
                Some(inter_x_) => inter_x = inter_x_,
                None => {
                    if abs_diff_eq!(inter_x, former.x()) {
                        interpolated.push((former.x(), former.y()));
                    }
                    break;
                }
            },
            (true, false) => {
                let inter_y = interpolate(inter_x, former, latter);

                interpolated.push((inter_x, inter_y));
            }
            (true, true) => match values_iter.next() {
                Some((former_, latter_)) => {
                    former = former_;
                    latter = latter_;
                }
                None => {
                    if abs_diff_eq!(inter_x, latter.x()) {
                        interpolated.push((latter.x(), latter.y()));
                    }
                    break;
                }
            },
            (false, true) => unreachable!(),
        }
    }

    interpolated
}

pub fn interpolate(inter_x: R64, left: impl AsXY<R64, R64>, right: impl AsXY<R64, R64>) -> R64 {
    let ldiff = inter_x - left.x();
    let rdiff = right.x() - inter_x;
    let diff = right.x() - left.x();

    (rdiff * left.y() + ldiff * right.y()) / diff
}

pub fn trapz<I, T>(values: impl IntoIterator<Item = T, IntoIter = I>) -> R64
where
    I: Iterator<Item = T> + Clone,
    T: AsXY<R64, R64>,
{
    let iter = values.into_iter();
    iter.clone()
        .zip(iter.skip(1))
        .map(|(left, right)| (left.y() + right.y()) * (right.x() - left.x()) / 2.0)
        .sum()
}
