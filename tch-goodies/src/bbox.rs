use crate::{
    common::*,
    ratio::Ratio,
    unit::{GridUnit, PixelUnit, RatioUnit, Unit},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LabeledBBox<T, U>
where
    U: Unit,
{
    pub bbox: BBox<T, U>,
    pub category_id: usize,
}

pub type LabeledPixelBBox<T> = LabeledBBox<T, PixelUnit>;
pub type LabeledGridBBox<T> = LabeledBBox<T, GridUnit>;
pub type LabeledRatioBBox = LabeledBBox<Ratio, RatioUnit>;

impl<U> LabeledBBox<R64, U>
where
    U: Unit,
{
    pub fn to_ratio_bbox(
        &self,
        image_height: usize,
        image_width: usize,
    ) -> Result<LabeledRatioBBox, LabeledRatioBBox> {
        let (bbox, has_violations) = match self.bbox.to_ratio_bbox(image_height, image_width) {
            Ok(bbox) => (bbox, false),
            Err(bbox) => (bbox, true),
        };
        let labeled_bbox = LabeledRatioBBox {
            bbox,
            category_id: self.category_id,
        };
        if !has_violations {
            Ok(labeled_bbox)
        } else {
            Err(labeled_bbox)
        }
    }
}

impl LabeledBBox<Ratio, RatioUnit> {
    pub fn crop(
        &self,
        top: Ratio,
        bottom: Ratio,
        left: Ratio,
        right: Ratio,
    ) -> Option<LabeledRatioBBox> {
        Some(LabeledRatioBBox {
            bbox: self.bbox.crop(top, bottom, left, right)?,
            category_id: self.category_id,
        })
    }
}

/// Bounding box in arbitrary units.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BBox<T, U>
where
    U: Unit,
{
    pub cycxhw: [T; 4],
    _phantom: PhantomData<U>,
}

pub type RatioBBox = BBox<Ratio, RatioUnit>;
pub type GridBBox<T> = BBox<T, GridUnit>;
pub type PixelBBox<T> = BBox<T, PixelUnit>;

impl<U> BBox<R64, U>
where
    U: Unit,
{
    pub fn try_from_tlbr(tlbr: [R64; 4]) -> Result<Self> {
        let [t, l, b, r] = tlbr;
        let cy = (t + b) / 2.0;
        let cx = (l + r) / 2.0;
        let h = b - t;
        let w = r - l;

        ensure!(h >= 0.0 && w >= 0.0, "invalid tlbr bbox {:?}", tlbr);

        Ok(Self {
            cycxhw: [cy, cx, h, w],
            _phantom: PhantomData,
        })
    }

    pub fn from_tlhw(tlhw: [R64; 4]) -> Self {
        let [t, l, h, w] = tlhw;
        let cy = t + h / 2.0;
        let cx = l + w / 2.0;

        Self {
            cycxhw: [cy, cx, h, w],
            _phantom: PhantomData,
        }
    }

    pub fn from_cycxhw(cycxhw: [R64; 4]) -> Self {
        Self {
            cycxhw,
            _phantom: PhantomData,
        }
    }

    pub fn tlbr(&self) -> [R64; 4] {
        let [cy, cx, h, w] = self.cycxhw;
        let t = cy - h / 2.0;
        let l = cx - w / 2.0;
        let b = cy + h / 2.0;
        let r = cx + w / 2.0;

        [t, l, b, r]
    }

    pub fn to_ratio_bbox(
        &self,
        image_height: usize,
        image_width: usize,
    ) -> Result<RatioBBox, RatioBBox> {
        let Self {
            cycxhw: [orig_cy, orig_cx, orig_h, orig_w],
            ..
        } = *self;

        let image_height = R64::new(image_height as f64);
        let image_width = R64::new(image_width as f64);

        let orig_t = orig_cy - orig_h / 2.0;
        let orig_l = orig_cx - orig_w / 2.0;
        let orig_b = orig_cy + orig_h / 2.0;
        let orig_r = orig_cx + orig_w / 2.0;

        // fix the value if it slightly exceeds the boundary
        let mut has_violations = false;

        let orig_t = if orig_t < 0.0 {
            has_violations = true;
            warn!("bbox top {} drops below the minimum 0.0", orig_t);
            R64::new(0.0)
        } else {
            orig_t
        };

        let orig_l = if orig_l < 0.0 {
            has_violations = true;
            warn!("bbox left {} drops below the minimum 0.0", orig_l);
            R64::new(0.0)
        } else {
            orig_l
        };

        let orig_b = if orig_b > image_height {
            has_violations = true;
            warn!(
                "bbox bottom {} exceeds the maximum {}",
                orig_b, image_height
            );
            image_height
        } else {
            orig_b
        };

        let orig_r = if orig_r > image_width {
            has_violations = true;
            warn!("bbox right {} exceeds the maximum {}", orig_r, image_width);
            image_width
        } else {
            orig_r
        };

        // construct ratio bbox
        let ratio_t = orig_t / image_height;
        let ratio_b = orig_b / image_height;
        let ratio_l = orig_l / image_width;
        let ratio_r = orig_r / image_width;

        let ratio_h = ratio_b - ratio_t;
        let ratio_w = ratio_r - ratio_l;
        let ratio_cy = ratio_t + ratio_h / 2.0;
        let ratio_cx = ratio_l + ratio_w / 2.0;

        let bbox = RatioBBox::new([
            ratio_cy.into(),
            ratio_cx.into(),
            ratio_h.into(),
            ratio_w.into(),
        ]);

        if !has_violations {
            Ok(bbox)
        } else {
            Err(bbox)
        }
    }
}

impl BBox<Ratio, RatioUnit> {
    pub fn new(cycxhw: [Ratio; 4]) -> Self {
        let [cy, cx, h, w] = cycxhw;

        // checked add, panic if overflow
        let _t = cy - h / 2.0;
        let _l = cx - w / 2.0;
        let _b = cy + h / 2.0;
        let _r = cx + w / 2.0;

        Self {
            cycxhw,
            _phantom: PhantomData,
        }
    }

    pub fn crop(&self, top: Ratio, bottom: Ratio, left: Ratio, right: Ratio) -> Option<RatioBBox> {
        let Self {
            cycxhw: [orig_cy, orig_cx, orig_h, orig_w],
            ..
        } = *self;

        let orig_t = orig_cy - orig_h / 2.0;
        let orig_l = orig_cx - orig_w / 2.0;
        let orig_b = orig_cy + orig_h / 2.0;
        let orig_r = orig_cx + orig_w / 2.0;

        let crop_t = orig_t.max(top).min(bottom);
        let crop_b = orig_b.max(top).min(bottom);
        let crop_l = orig_l.max(left).min(right);
        let crop_r = orig_r.max(left).min(right);

        if abs_diff_eq!(crop_t.raw(), crop_b.raw()) || abs_diff_eq!(crop_l.raw(), crop_r.raw()) {
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

    pub fn to_r64_bbox<U>(&self, height: R64, width: R64) -> BBox<R64, U>
    where
        U: Unit,
    {
        let Self {
            cycxhw: [ratio_cy, ratio_cx, ratio_h, ratio_w],
            ..
        } = *self;

        let cy = *ratio_cy * height;
        let cx = *ratio_cx * width;
        let h = *ratio_h * height;
        let w = *ratio_w * width;

        BBox {
            cycxhw: [cy, cx, h, w],
            _phantom: PhantomData,
        }
    }

    /// Returns range in tlbr format.
    pub fn tlbr(&self) -> [Ratio; 4] {
        let [cy, cx, h, w] = self.cycxhw;
        let t = cy - h / 2.0;
        let l = cx - w / 2.0;
        let b = cy + h / 2.0;
        let r = cx + w / 2.0;

        [t, l, b, r]
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

        if abs_diff_eq!(cy.raw(), 0.0) || abs_diff_eq!(cx.raw(), 0.0) {
            return None;
        }

        Some([cy, cx, h, w])
    }
}
