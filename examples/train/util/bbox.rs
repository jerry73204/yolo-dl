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
            tlhw: [orig_t, orig_l, orig_h, orig_w],
            category_id,
        } = *self;

        let image_height = R64::new(image_height as f64);
        let image_width = R64::new(image_width as f64);

        let mut orig_b = orig_t + orig_h;
        let mut orig_r = orig_l + orig_w;

        // fix the value if it slightly exceeds the boundary
        if orig_b > image_height {
            if abs_diff_eq!(orig_b.raw(), image_height.raw()) {
                orig_b = image_height;
            } else {
                panic!("the bbox exceeds the image boundary");
            }
        }

        if orig_r > image_width {
            if abs_diff_eq!(orig_r.raw(), image_width.raw()) {
                orig_r = image_width;
            } else {
                panic!("the bbox exceeds the image boundary");
            }
        }

        // construct ratio bbox
        let new_t = orig_t / image_height;
        let new_b = orig_b / image_height;
        let new_l = orig_l / image_width;
        let new_r = orig_r / image_width;

        let new_h = new_b - new_t;
        let new_w = new_r - new_l;

        RatioBBox::new(
            [new_t.into(), new_l.into(), new_h.into(), new_w.into()],
            category_id,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RatioBBox {
    pub tlhw: [Ratio; 4],
    pub category_id: usize,
}

impl RatioBBox {
    pub fn new(tlhw: [Ratio; 4], category_id: usize) -> Self {
        let [t, l, h, w] = tlhw;

        // checked add
        let _b = t + h;
        let _r = l + w;

        Self { tlhw, category_id }
    }

    pub fn crop(&self, top: Ratio, bottom: Ratio, left: Ratio, right: Ratio) -> Option<RatioBBox> {
        let Self {
            tlhw: [orig_t, orig_l, orig_h, orig_w],
            category_id,
        } = *self;

        let orig_b = orig_t + orig_h;
        let orig_r = orig_l + orig_w;

        let crop_t = orig_t.max(top);
        let crop_b = orig_b.max(bottom);
        let crop_l = orig_l.max(left);
        let crop_r = orig_r.max(right);

        if crop_l < crop_r && crop_t < crop_b {
            let crop_w = crop_r - crop_l;
            let crop_h = crop_b - crop_t;

            Some(RatioBBox {
                tlhw: [crop_t, crop_l, crop_h, crop_w],
                category_id,
            })
        } else {
            None
        }
    }
}
