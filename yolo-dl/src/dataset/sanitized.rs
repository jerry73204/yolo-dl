use super::*;
use crate::common::*;

/// The dataset that filters out bad boxes.
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

        let mut filtered_bbox_count = 0;

        let records: Vec<_> = dataset
            .records()
            .iter()
            .map(|record| -> Result<_> {
                let FileRecord {
                    ref path,
                    ref size,
                    bboxes: ref orig_bboxes,
                } = *record.as_ref();

                ensure!(
                    size.h > 0 && size.w > 0,
                    "image height and width must be positive"
                );

                let range_h = (-out_of_bound_tolerance)..(out_of_bound_tolerance + size.h as f64);
                let range_w = (-out_of_bound_tolerance)..(out_of_bound_tolerance + size.w as f64);

                let bboxes: Vec<_> = orig_bboxes
                    .iter()
                    .map(|bbox| -> Result<_> {
                        let PixelRectLabel {
                            rect: ref cycxhw,
                            class,
                        } = *bbox;
                        let tlbr: PixelTLBR<_> = cycxhw.into();

                        // out of bound check with tolerance
                        ensure!(
                            range_h.contains(&tlbr.t())
                                && range_h.contains(&tlbr.b())
                                && range_w.contains(&tlbr.l())
                                && range_w.contains(&tlbr.r()),
                            "bbox {:?} range out of bound with out_of_bound_tolerance {}",
                            cycxhw,
                            out_of_bound_tolerance
                        );

                        // crop out out of bound parts
                        let sanitized_t = clamp(tlbr.t(), r64(0.0), r64(size.h as f64));
                        let sanitized_b = clamp(tlbr.b(), r64(0.0), r64(size.h as f64));
                        let sanitized_l = clamp(tlbr.l(), r64(0.0), r64(size.w as f64));
                        let sanitized_r = clamp(tlbr.r(), r64(0.0), r64(size.w as f64));

                        debug_assert!(
                            (0.0..=(size.h as f64)).contains(&sanitized_t.raw())
                                && (0.0..=(size.h as f64)).contains(&sanitized_b.raw())
                                && (0.0..=(size.w as f64)).contains(&sanitized_l.raw())
                                && (0.0..=(size.w as f64)).contains(&sanitized_r.raw())
                        );

                        // kick of small bboxes
                        let sanitized_h = sanitized_b - sanitized_t;
                        let sanitized_w = sanitized_r - sanitized_l;

                        if sanitized_h / size.h as f64 <= min_bbox_size
                            || sanitized_w / size.w as f64 <= min_bbox_size
                        {
                            return Ok(None);
                        }

                        // save sanitized bbox
                        let sanitized_bbox = PixelRectLabel {
                            rect: PixelCyCxHW::from_tlbr(
                                sanitized_t,
                                sanitized_l,
                                sanitized_b,
                                sanitized_r,
                            )
                            .unwrap(),
                            class,
                        };

                        Ok(Some(sanitized_bbox))
                    })
                    .filter_map(|result| result.transpose())
                    .try_collect()
                    .with_context(|| format!("fail to open file '{}'", path.display()))?;

                filtered_bbox_count += orig_bboxes.len() - bboxes.len();

                Ok(Arc::new(FileRecord {
                    path: path.clone(),
                    size: size.clone(),
                    bboxes,
                }))
            })
            .try_collect()?;

        if filtered_bbox_count > 0 {
            warn!(
                "filtered out {} bad objects in the data set",
                filtered_bbox_count
            );
        }

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
