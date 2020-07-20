use crate::{common::*, util::Ratio};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PixelBBox {
    pub tlhw: [R64; 4],
    pub category_id: usize,
}

impl PixelBBox {
    pub fn new(tlhw: [R64; 4], category_id: usize) -> Self {
        let [t, l, h, w] = tlhw;
        assert!(t >= 0.0);
        assert!(l >= 0.0);
        assert!(h >= 0.0);
        assert!(w >= 0.0);

        Self { tlhw, category_id }
    }

    pub fn to_ratio_bbox(&self, image_height: usize, image_width: usize) -> RatioBBox {
        let Self {
            tlhw: [pixel_t, pixel_l, pixel_h, pixel_w],
            category_id,
        } = *self;

        let image_height = R64::new(image_height as f64);
        let image_width = R64::new(image_width as f64);

        let mut pixel_b = pixel_t + pixel_h;
        let mut pixel_r = pixel_l + pixel_w;

        // fix the value if it slightly exceeds the boundary
        if pixel_b > image_height {
            if abs_diff_eq!(pixel_b.raw(), image_height.raw()) {
                pixel_b = image_height;
            } else {
                panic!("the bbox exceeds the image boundary");
            }
        }

        if pixel_r > image_width {
            if abs_diff_eq!(pixel_r.raw(), image_width.raw()) {
                pixel_r = image_width;
            } else {
                panic!("the bbox exceeds the image boundary");
            }
        }

        // construct ratio bbox
        let ratio_t = pixel_t / image_height;
        let ratio_b = pixel_b / image_height;
        let ratio_l = pixel_l / image_width;
        let ratio_r = pixel_r / image_width;

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
