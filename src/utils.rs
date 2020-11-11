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
