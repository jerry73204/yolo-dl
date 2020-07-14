use crate::common::*;

pub fn timed<F, T>(mut f: F) -> (T, Instant)
where
    F: FnMut() -> T,
{
    let instant = Instant::now();
    let ret = f();
    (ret, instant)
}

pub fn try_timed<F, T, E>(mut f: F) -> Result<(T, Instant), E>
where
    F: FnMut() -> Result<T, E>,
{
    let instant = Instant::now();
    let ret = f()?;
    Ok((ret, instant))
}

pub async fn timed_async<Fut, T>(f: Fut) -> (T, Instant)
where
    Fut: Future<Output = T>,
{
    let instant = Instant::now();
    let ret = f.await;
    (ret, instant)
}

pub async fn try_timed_async<Fut, T, E>(f: Fut) -> Result<(T, Instant), E>
where
    Fut: Future<Output = Result<T, E>>,
{
    let instant = Instant::now();
    let ret = f.await?;
    Ok((ret, instant))
}
