use super::*;
use crate::common::*;

#[derive(Debug)]
pub struct SanitizedDataset<D>
where
    D: FileDataset,
{
    dataset: D,
    records: Vec<Arc<FileRecord>>,
}

impl<D> SanitizedDataset<D>
where
    D: FileDataset,
{
    pub fn new(dataset: D, out_of_bound_tolerance: R64, min_bbox_size: Ratio) -> Result<Self> {
        let min_bbox_size = min_bbox_size.to_r64();
        ensure!(
            out_of_bound_tolerance >= 0.0,
            "out_of_bound_tolerance must be non-negative"
        );

        let records: Vec<_> = dataset
            .records()
            .iter()
            .map(|record| -> Result<_> {
                let FileRecord {
                    ref path,
                    size: PixelSize { height, width, .. },
                    ref bboxes,
                } = *record.as_ref();

                ensure!(
                    height > 0 && width > 0,
                    "image height and width must be positive"
                );

                let range_h = (-out_of_bound_tolerance)..(out_of_bound_tolerance + height as f64);
                let range_w = (-out_of_bound_tolerance)..(out_of_bound_tolerance + width as f64);

                let bboxes: Vec<_> = bboxes
                    .iter()
                    .map(|bbox| -> Result<_> {
                        let LabeledPixelBBox {
                            ref bbox,
                            category_id,
                        } = *bbox;
                        let [orig_t, orig_l, orig_b, orig_r] = bbox.tlbr();

                        // out of bound check with tolerance
                        ensure!(
                            range_h.contains(&orig_t)
                                && range_h.contains(&orig_b)
                                && range_w.contains(&orig_l)
                                && range_w.contains(&orig_r),
                            "bbox {:?} range out of bound with out_of_bound_tolerance {}",
                            bbox,
                            out_of_bound_tolerance
                        );

                        // crop out out of bound parts
                        let sanitized_t = clamp(orig_t, R64::new(0.0), R64::new(height as f64));
                        let sanitized_b = clamp(orig_b, R64::new(0.0), R64::new(height as f64));
                        let sanitized_l = clamp(orig_l, R64::new(0.0), R64::new(width as f64));
                        let sanitized_r = clamp(orig_r, R64::new(0.0), R64::new(width as f64));

                        debug_assert!(
                            (0.0..=(height as f64)).contains(&sanitized_t.raw())
                                && (0.0..=(height as f64)).contains(&sanitized_b.raw())
                                && (0.0..=(width as f64)).contains(&sanitized_l.raw())
                                && (0.0..=(width as f64)).contains(&sanitized_r.raw())
                        );

                        // kick of small bboxes
                        let sanitized_h = sanitized_b - sanitized_t;
                        let sanitized_w = sanitized_r - sanitized_l;

                        if sanitized_h / height as f64 <= min_bbox_size
                            || sanitized_w / width as f64 <= min_bbox_size
                        {
                            warn!("filtered out a small bbox from '{}'", path.display());
                            return Ok(None);
                        }

                        // save sanitized bbox
                        let sanitized_bbox = LabeledPixelBBox {
                            bbox: PixelBBox::try_from_tlbr([
                                sanitized_t,
                                sanitized_l,
                                sanitized_b,
                                sanitized_r,
                            ])
                            .unwrap(),
                            category_id,
                        };

                        Ok(Some(sanitized_bbox))
                    })
                    .filter_map(|result| result.transpose())
                    .try_collect()?;

                Ok(Arc::new(FileRecord {
                    path: path.clone(),
                    size: PixelSize::new(height, width),
                    bboxes,
                }))
            })
            .try_collect()?;

        Ok(Self { dataset, records })
    }
}

impl<D> GenericDataset for SanitizedDataset<D>
where
    D: FileDataset,
{
    fn input_channels(&self) -> usize {
        self.dataset.input_channels()
    }

    fn classes(&self) -> &IndexSet<String> {
        self.dataset.classes()
    }
}

impl<D> FileDataset for SanitizedDataset<D>
where
    D: FileDataset,
{
    fn records(&self) -> &[Arc<FileRecord>] {
        &self.records
    }
}

fn clamp(value: R64, min: R64, max: R64) -> R64 {
    value.max(min).min(max)
}
