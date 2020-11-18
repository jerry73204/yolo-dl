use crate::common::*;

pub fn transpose_matrix<T>(buf: &mut [T], nrows: usize, ncols: usize) -> Result<()>
where
    T: Clone,
{
    ensure!(buf.len() == nrows * ncols, "the size does not match");
    let tmp = buf.to_owned();

    (0..nrows).for_each(|row| {
        (0..ncols).for_each(|col| {
            buf[col * nrows + row] = tmp[row * ncols + col].clone();
        });
    });

    Ok(())
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct DisplayAsDebug<T>(pub T)
where
    T: Display;

impl<T> Debug for DisplayAsDebug<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

unzip_n!(pub 2);
unzip_n!(pub 3);
unzip_n!(pub 7);
