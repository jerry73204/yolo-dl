//! Safe bounding box types and functions.

use crate::{
    common::*,
    ratio::Ratio,
    unit::{GridUnit, PixelUnit, RatioUnit, Unit, Unitless},
};

pub use bbox::*;
pub use labeled_bbox::*;

mod labeled_bbox {
    use super::*;

    /// Generic bounding box with an extra class ID.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct LabeledBBox<T, U>
    where
        T: Num + Copy,
        U: Unit,
    {
        pub bbox: BBox<T, U>,
        pub category_id: usize,
    }

    pub type LabeledPixelBBox<T> = LabeledBBox<T, PixelUnit>;
    pub type LabeledGridBBox<T> = LabeledBBox<T, GridUnit>;
    pub type LabeledRatioBBox = LabeledBBox<Ratio, RatioUnit>;

    impl<T, U> LabeledBBox<T, U>
    where
        T: Num + Copy,
        U: Unit,
    {
        pub fn cycxhw(&self) -> [T; 4] {
            self.bbox.cycxhw()
        }

        pub fn tlbr(&self) -> [T; 4]
        where
            T: Num + Copy,
        {
            self.bbox.tlbr()
        }

        pub fn map_elem<F, R>(&self, f: F) -> LabeledBBox<R, U>
        where
            F: FnMut(T) -> R,
            R: Num + Copy,
        {
            let Self {
                ref bbox,
                category_id,
            } = *self;

            LabeledBBox {
                bbox: bbox.map_elem(f),
                category_id,
            }
        }

        // pub fn corners(&self) -> Corners<T, U>
        // where
        //     T: Num,
        // {
        //     self.bbox.corners()
        // }

        pub fn to_unit<V>(&self, h_scale: T, w_scale: T) -> LabeledBBox<T, V>
        where
            V: Unit,
        {
            let Self {
                ref bbox,
                category_id,
            } = *self;

            LabeledBBox {
                bbox: bbox.to_unit(h_scale, w_scale),
                category_id,
            }
        }
    }

    impl<U> LabeledBBox<R64, U>
    where
        U: Unit,
    {
        pub fn scale(&self, scale: R64) -> Self {
            let Self {
                ref bbox,
                category_id,
            } = *self;
            Self {
                bbox: bbox.scale(scale),
                category_id,
            }
        }

        pub fn to_ratio_bbox(
            &self,
            image_height: usize,
            image_width: usize,
        ) -> Result<LabeledRatioBBox> {
            let bbox = self
                .bbox
                .to_ratio_bbox(R64::new(image_height as f64), R64::new(image_width as f64))?;
            let labeled_bbox = LabeledRatioBBox {
                bbox,
                category_id: self.category_id,
            };
            Ok(labeled_bbox)
        }
    }

    impl LabeledBBox<Ratio, RatioUnit> {
        pub fn scale(&self, scale: R64) -> Self {
            let Self {
                ref bbox,
                category_id,
            } = *self;
            Self {
                bbox: bbox.scale(scale),
                category_id,
            }
        }

        pub fn crop(&self, tlbr: [Ratio; 4]) -> Option<LabeledRatioBBox> {
            Some(LabeledRatioBBox {
                bbox: self.bbox.crop(tlbr)?,
                category_id: self.category_id,
            })
        }

        pub fn to_r64_bbox<U>(&self, height: usize, width: usize) -> LabeledBBox<R64, U>
        where
            U: Unit,
        {
            let Self {
                ref bbox,
                category_id,
            } = *self;
            LabeledBBox {
                bbox: bbox.to_r64_bbox(height, width),
                category_id,
            }
        }
    }

    impl<T, U> AsRef<BBox<T, U>> for LabeledBBox<T, U>
    where
        T: Num + Copy,
        U: Unit,
    {
        fn as_ref(&self) -> &BBox<T, U> {
            &self.bbox
        }
    }
}

mod bbox {
    use super::*;

    /// Bounding box in arbitrary units.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct BBox<T, U>
    where
        T: Num + Copy,
        U: Unit,
    {
        cycxhw: [T; 4],
        _phantom: PhantomData<U>,
    }

    pub type RatioBBox = BBox<Ratio, RatioUnit>;
    pub type GridBBox<T> = BBox<T, GridUnit>;
    pub type PixelBBox<T> = BBox<T, PixelUnit>;
    pub type UnitlessBBox<T> = BBox<T, Unitless>;

    impl<T, U> BBox<T, U>
    where
        T: Num + Copy,
        U: Unit,
    {
        pub fn tlbr(&self) -> [T; 4] {
            let two = T::one() + T::one();
            let [cy, cx, h, w] = self.cycxhw;
            let t = cy - h / two;
            let l = cx - w / two;
            let b = cy + h / two;
            let r = cx + w / two;
            [t, l, b, r]
        }

        pub fn cycxhw(&self) -> [T; 4] {
            self.cycxhw
        }

        pub fn map_elem<F, R>(&self, mut f: F) -> BBox<R, U>
        where
            F: FnMut(T) -> R,
            R: Num + Copy,
        {
            let [cy, cx, h, w] = self.cycxhw;
            BBox {
                cycxhw: [f(cy), f(cx), f(h), f(w)],
                _phantom: PhantomData,
            }
        }

        pub fn to_unit<V>(&self, h_scale: T, w_scale: T) -> BBox<T, V>
        where
            V: Unit,
        {
            let [cy, cx, h, w] = self.cycxhw;
            BBox {
                cycxhw: [cy * h_scale, cx * w_scale, h * h_scale, w * w_scale],
                _phantom: PhantomData,
            }
        }
    }

    impl<U> BBox<f64, U>
    where
        U: Unit,
    {
        pub fn try_from_cycxhw(cycxhw: [f64; 4]) -> Result<Self> {
            let [_cy, _cx, h, w] = cycxhw;
            ensure!(h >= 0.0 && w >= 0.0, "invalid cycxhw {:?}", cycxhw);
            Ok(Self {
                cycxhw,
                _phantom: PhantomData,
            })
        }
    }

    impl<U> BBox<R64, U>
    where
        U: Unit,
    {
        pub fn try_from_tlbr(tlbr: [R64; 4]) -> Result<Self> {
            let [t, l, b, r] = tlbr;
            ensure!(t <= b && l <= r, "invalid tlbr {:?}", tlbr);

            let cy = (t + b) / 2.0;
            let cx = (l + r) / 2.0;
            let h = b - t;
            let w = r - l;

            Ok(Self {
                cycxhw: [cy, cx, h, w],
                _phantom: PhantomData,
            })
        }

        pub fn try_from_tlhw(tlhw: [R64; 4]) -> Result<Self> {
            let [t, l, h, w] = tlhw;
            ensure!(h >= 0.0 && w >= 0.0, "invalid tlhw {:?}", tlhw);
            let cy = t + h / 2.0;
            let cx = l + w / 2.0;

            Ok(Self {
                cycxhw: [cy, cx, h, w],
                _phantom: PhantomData,
            })
        }

        pub fn try_from_cycxhw(cycxhw: [R64; 4]) -> Result<Self> {
            let [_cy, _cx, h, w] = cycxhw;
            ensure!(h >= 0.0 && w >= 0.0, "invalid cycxhw {:?}", cycxhw);
            Ok(Self {
                cycxhw,
                _phantom: PhantomData,
            })
        }

        pub fn to_ratio_bbox(&self, max_height: R64, max_width: R64) -> Result<RatioBBox> {
            // construct ratio bbox
            let [orig_cy, orig_cx, orig_h, orig_w] = self.cycxhw;

            let ratio_cy = Ratio::try_from(orig_cy / max_height)?;
            let ratio_cx = Ratio::try_from(orig_cx / max_width)?;
            let ratio_h = Ratio::try_from(orig_h / max_height)?;
            let ratio_w = Ratio::try_from(orig_w / max_width)?;

            RatioBBox::try_from_cycxhw([ratio_cy, ratio_cx, ratio_h, ratio_w])
        }

        pub fn scale(&self, scale: R64) -> Self {
            let [cy, cx, h, w] = self.cycxhw;
            Self {
                cycxhw: [cy, cx, h * scale, w * scale],
                _phantom: PhantomData,
            }
        }
    }

    impl RatioBBox {
        pub fn try_from_cycxhw(cycxhw: [Ratio; 4]) -> Result<Self> {
            let [cy, cx, h, w] = cycxhw;

            // verify boundary
            let _ratio_t = cy.checked_sub(h / 2.0)?;
            let _ratio_l = cx.checked_sub(w / 2.0)?;
            let _ratio_b = cy.checked_add(h / 2.0)?;
            let _ratio_r = cx.checked_add(w / 2.0)?;

            Ok(Self {
                cycxhw: [cy, cx, h, w],
                _phantom: PhantomData,
            })
        }

        pub fn try_from_tlbr(tlbr: [Ratio; 4]) -> Result<Self> {
            let [t, l, b, r] = tlbr;
            let h = b.checked_sub(t)?;
            let w = r.checked_sub(l)?;
            let cy = t.checked_add(h / 2.0)?;
            let cx = l.checked_add(w / 2.0)?;
            Self::try_from_cycxhw([cy, cx, h, w])
        }

        pub fn crop(&self, tlbr: [Ratio; 4]) -> Option<RatioBBox> {
            let Self {
                cycxhw: [orig_cy, orig_cx, orig_h, orig_w],
                ..
            } = *self;

            let [margin_t, margin_l, margin_b, margin_r] = tlbr;

            let orig_t = orig_cy - orig_h / 2.0;
            let orig_l = orig_cx - orig_w / 2.0;
            let orig_b = orig_cy + orig_h / 2.0;
            let orig_r = orig_cx + orig_w / 2.0;

            let crop_t = orig_t.max(margin_t).min(margin_b);
            let crop_b = orig_b.max(margin_t).min(margin_b);
            let crop_l = orig_l.max(margin_l).min(margin_r);
            let crop_r = orig_r.max(margin_l).min(margin_r);

            if abs_diff_eq!(crop_t, crop_b) || abs_diff_eq!(crop_l, crop_r) {
                None
            } else {
                let crop_h = crop_b - crop_t;
                let crop_w = crop_r - crop_l;
                let crop_cy = crop_t + crop_h / 2.0;
                let crop_cx = crop_l + crop_w / 2.0;

                Some(RatioBBox {
                    cycxhw: [crop_cy, crop_cx, crop_h, crop_w],
                    _phantom: PhantomData,
                })
            }
        }

        pub fn to_r64_bbox<U>(&self, height: usize, width: usize) -> BBox<R64, U>
        where
            U: Unit,
        {
            let height = R64::new(height as f64);
            let width = R64::new(width as f64);
            let Self {
                cycxhw: [ratio_cy, ratio_cx, ratio_h, ratio_w],
                ..
            } = *self;

            let cy = ratio_cy.to_r64() * height;
            let cx = ratio_cx.to_r64() * width;
            let h = ratio_h.to_r64() * height;
            let w = ratio_w.to_r64() * width;

            BBox {
                cycxhw: [cy, cx, h, w],
                _phantom: PhantomData,
            }
        }

        pub fn scale(&self, scale: R64) -> Self {
            let Self {
                cycxhw: [orig_cy, orig_cx, orig_h, orig_w],
                ..
            } = *self;

            let tmp_h = orig_h.to_r64() * scale.max(R64::new(1.0));
            let tmp_w = orig_w.to_r64() * scale.max(R64::new(1.0));
            let new_t = (orig_cy.to_r64() - tmp_h / 2.0).max(R64::new(0.0));
            let new_b = (orig_cy.to_r64() + tmp_h / 2.0).min(R64::new(1.0));
            let new_l = (orig_cx.to_r64() - tmp_w / 2.0).max(R64::new(0.0));
            let new_r = (orig_cx.to_r64() + tmp_w / 2.0).min(R64::new(1.0));
            let new_cy = (new_t + new_b) / 2.0;
            let new_cx = (new_l + new_r) / 2.0;
            let new_h = new_b - new_t;
            let new_w = new_r - new_l;
            let cycxhw = [
                new_cy.try_into().unwrap(),
                new_cx.try_into().unwrap(),
                new_h.try_into().unwrap(),
                new_w.try_into().unwrap(),
            ];

            Self {
                cycxhw,
                _phantom: PhantomData,
            }
        }

        /// Compute intersection area in cycxhw format.
        pub fn intersect(&self, rhs: &Self) -> Option<[Ratio; 4]> {
            let [lt, ll, lb, lr] = self.tlbr();
            let [rt, rl, rb, rr] = rhs.tlbr();

            let t = lt.max(rt);
            let l = ll.max(rl);
            let b = lb.min(rb);
            let r = lr.min(rr);

            let h = b - t;
            let w = r - l;
            let cy = t + h / 2.0;
            let cx = l + w / 2.0;

            if abs_diff_eq!(cy, 0.0) || abs_diff_eq!(cx, 0.0) {
                return None;
            }

            Some([cy, cx, h, w])
        }
    }

    impl<T, U> AsRef<BBox<T, U>> for BBox<T, U>
    where
        T: Num + Copy,
        U: Unit,
    {
        fn as_ref(&self) -> &BBox<T, U> {
            &self
        }
    }
}
