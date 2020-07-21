use crate::{common::*, utils::Ratio};

/// Bounding box in arbitrary units.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BBox {
    pub cycxhw: [R64; 4],
    pub category_id: usize,
}

impl BBox {
    pub fn from_tlhw(tlhw: [R64; 4], category_id: usize) -> Self {
        let [t, l, h, w] = tlhw;
        let cy = t + h / 2.0;
        let cx = l + w / 2.0;

        Self {
            cycxhw: [cy, cx, h, w],
            category_id,
        }
    }

    pub fn from_cycxhw(cycxhw: [R64; 4], category_id: usize) -> Self {
        Self {
            cycxhw,
            category_id,
        }
    }

    pub fn to_ratio_bbox(&self, image_height: usize, image_width: usize) -> RatioBBox {
        let Self {
            cycxhw: [orig_cy, orig_cx, orig_h, orig_w],
            category_id,
        } = *self;

        let image_height = R64::new(image_height as f64);
        let image_width = R64::new(image_width as f64);

        let mut orig_t = orig_cy - orig_h / 2.0;
        let mut orig_l = orig_cx - orig_w / 2.0;
        let mut orig_b = orig_cy + orig_h / 2.0;
        let mut orig_r = orig_cx + orig_w / 2.0;

        // fix the value if it slightly exceeds the boundary
        if orig_t < 0.0 {
            if abs_diff_eq!(orig_t.raw(), 0.0) {
                orig_t = R64::new(0.0);
            } else {
                panic!(
                    "the bbox exceeds the image boundary: expect minimum top 0.0, found {}",
                    orig_t
                );
            }
        }

        if orig_l < 0.0 {
            if abs_diff_eq!(orig_l.raw(), 0.0) {
                orig_l = R64::new(0.0);
            } else {
                panic!(
                    "the bbox exceeds the image boundary: expect minimum left 0.0, found {}",
                    orig_l
                );
            }
        }

        if orig_b > image_height {
            if abs_diff_eq!(orig_b.raw(), image_height.raw()) {
                orig_b = image_height;
            } else {
                panic!(
                    "the bbox exceeds the image boundary: expect maximum bottom {}, found {}",
                    image_height, orig_b
                );
            }
        }

        if orig_r > image_width {
            if abs_diff_eq!(orig_r.raw(), image_width.raw()) {
                orig_r = image_width;
            } else {
                panic!(
                    "the bbox exceeds the image boundary: expect maximum right {}, found {}",
                    image_width, orig_r
                );
            }
        }

        // construct ratio bbox
        let ratio_t = orig_t / image_height;
        let ratio_b = orig_b / image_height;
        let ratio_l = orig_l / image_width;
        let ratio_r = orig_r / image_width;

        let ratio_h = ratio_b - ratio_t;
        let ratio_w = ratio_r - ratio_l;
        let ratio_cy = ratio_t + ratio_h / 2.0;
        let ratio_cx = ratio_l + ratio_w / 2.0;

        RatioBBox::new(
            [
                ratio_cy.into(),
                ratio_cx.into(),
                ratio_h.into(),
                ratio_w.into(),
            ],
            category_id,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RatioBBox {
    pub cycxhw: [Ratio; 4],
    pub category_id: usize,
}

impl RatioBBox {
    pub fn new(cycxhw: [Ratio; 4], category_id: usize) -> Self {
        let [cy, cx, h, w] = cycxhw;

        // checked add
        let _t = cy - h / 2.0;
        let _l = cx - w / 2.0;
        let _b = cy + h / 2.0;
        let _r = cx + w / 2.0;

        Self {
            cycxhw,
            category_id,
        }
    }

    pub fn crop(&self, top: Ratio, bottom: Ratio, left: Ratio, right: Ratio) -> Option<RatioBBox> {
        let Self {
            cycxhw: [orig_cy, orig_cx, orig_h, orig_w],
            category_id,
        } = *self;

        let orig_t = orig_cy - orig_h / 2.0;
        let orig_l = orig_cx - orig_w / 2.0;
        let orig_b = orig_cy + orig_h / 2.0;
        let orig_r = orig_cx + orig_w / 2.0;

        let crop_t = orig_t.max(top);
        let crop_b = orig_b.max(bottom);
        let crop_l = orig_l.max(left);
        let crop_r = orig_r.max(right);

        if crop_l < crop_r && crop_t < crop_b {
            let crop_w = crop_r - crop_l;
            let crop_h = crop_b - crop_t;
            let crop_cy = crop_t + crop_h / 2.0;
            let crop_cx = crop_l + crop_w / 2.0;

            Some(RatioBBox {
                cycxhw: [crop_cy, crop_cx, crop_h, crop_w],
                category_id,
            })
        } else {
            None
        }
    }

    pub fn to_bbox(&self, height: R64, width: R64) -> BBox {
        let Self {
            cycxhw: [ratio_cy, ratio_cx, ratio_h, ratio_w],
            category_id,
        } = *self;

        let cy = *ratio_cy * height;
        let cx = *ratio_cx * width;
        let h = *ratio_h * height;
        let w = *ratio_w * width;

        BBox {
            cycxhw: [cy, cx, h, w],
            category_id,
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
        let l = lt.max(rt);
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
