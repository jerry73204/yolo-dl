use crate::common::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HW<T> {
    w: T,
    h: T,
}
impl<T> HW<T> {
    pub fn try_cast<U>(self) -> Option<HW<U>>
    where
        T: ToPrimitive,
        U: NumCast,
    {
        Some(HW {
            h: U::from(self.h)?,
            w: U::from(self.w)?,
        })
    }

    pub fn cast<U>(self) -> HW<U>
    where
        T: ToPrimitive,
        U: NumCast,
    {
        self.try_cast().unwrap()
    }
}

impl<T> HW<T>
where
    T: Num + PartialOrd + Copy,
{
    pub fn try_from_hw(hw: [T; 2]) -> Result<Self> {
        let [h, w] = hw;
        let zero = T::zero();
        ensure!(
            h >= zero && w >= zero,
            "height and width parameters must be non-negative"
        );
        Ok(Self { w, h })
    }

    pub fn from_hw(hw: [T; 2]) -> Self {
        Self::try_from_hw(hw).unwrap()
    }

    pub fn area(&self) -> T {
        self.w * self.h
    }

    /// Get a reference to the size's w.
    pub fn w(&self) -> T {
        self.w
    }

    /// Get a reference to the size's h.
    pub fn h(&self) -> T {
        self.h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn size_area() {
        let s1 = HW::from_hw([3.0, 2.0]);
        let area: f64 = s1.area();
        assert_abs_diff_eq!(area, 6.0);
    }
}
