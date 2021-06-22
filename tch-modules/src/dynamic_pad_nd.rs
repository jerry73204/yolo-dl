use crate::common::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PaddingKind {
    Zero,
    Replication,
    Reflection,
}

#[derive(Debug)]
pub struct DynamicPad<const N: usize> {
    padding: Vec<i64>,
    kind: PaddingKind,
}

impl<const N: usize> DynamicPad<N> {
    pub fn new(kind: PaddingKind, padding: &[usize]) -> Result<Self> {
        match (kind, N) {
            (PaddingKind::Replication, 1 | 2 | 3)
            | (PaddingKind::Reflection | PaddingKind::Zero, 1 | 2) => {}
            _ => bail!("{} dimensional {:?} padding is not supported", N, kind),
        }

        ensure!(
            padding.len() == N * 2,
            "expect padding slice length {}, but get {}",
            N * 2,
            padding.len()
        );

        let padding: Vec<_> = padding.iter().map(|&pad| pad as i64).collect();

        Ok(Self { padding, kind })
    }
}

impl<const N: usize> nn::Module for DynamicPad<N> {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match (N, self.kind) {
            (1, PaddingKind::Zero) => match *self.padding {
                [l, r] => xs.zero_pad1d(l, r),
                _ => unreachable!(),
            },
            (2, PaddingKind::Zero) => match *self.padding {
                [l, r, t, b] => xs.zero_pad2d(l, r, t, b),
                _ => unreachable!(),
            },
            (1, PaddingKind::Reflection) => xs.reflection_pad1d(&self.padding),
            (2, PaddingKind::Reflection) => xs.reflection_pad2d(&self.padding),
            (1, PaddingKind::Replication) => xs.replication_pad1d(&self.padding),
            (2, PaddingKind::Replication) => xs.replication_pad2d(&self.padding),
            (3, PaddingKind::Replication) => xs.replication_pad3d(&self.padding),
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_pad_test() {
        let pad = DynamicPad::<2>::new(PaddingKind::Reflection, &[1, 2, 3, 4]).unwrap();
        let input = Tensor::randn(&[4, 3, 8, 8], tch::kind::FLOAT_CPU);
        let output = pad.forward(&input);
        assert_eq!(output.size(), [4, 3, 15, 11]);
    }
}
