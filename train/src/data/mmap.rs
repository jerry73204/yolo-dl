use super::*;
use crate::common::*;
use mmap_dataset::{BBoxEntry, ComponentKind, Header};

#[derive(Debug)]
pub struct MmapDataset {
    dataset: Arc<mmap_dataset::Dataset>,
    device: Device,
}

impl MmapDataset {
    pub async fn load(path: impl AsRef<async_std::path::Path>, device: Device) -> Result<Self> {
        let dataset = Arc::new(mmap_dataset::Dataset::open(path).await?);
        Ok(Self { dataset, device })
    }
}

impl GenericDataset for MmapDataset {
    fn input_channels(&self) -> usize {
        let [channels, _h, _w] = self.dataset.header.shape;
        channels as usize
    }

    fn classes(&self) -> &IndexSet<String> {
        &self.dataset.classes
    }
}

impl RandomAccessDataset for MmapDataset {
    fn num_records(&self) -> usize {
        self.dataset.num_images()
    }

    fn nth(&self, index: usize) -> Pin<Box<dyn Future<Output = Result<DataRecord>> + Send>> {
        let dataset = self.dataset.clone();
        let device = self.device;

        Box::pin(async move {
            let mut timing = Timing::new("mmap dataset nth");

            let mmap_dataset::Dataset {
                header:
                    Header {
                        shape: [channels, height, width],
                        component_kind,
                        ..
                    },
                ..
            } = *dataset;

            let (image, bboxes, timing) = async_std::task::spawn_blocking(move || -> Result<_> {
                let (image, bboxes) = dataset
                    .nth(index)
                    .ok_or_else(|| format_err!("invalid index {}", index))?;
                let image = image.to_device(device);

                timing.set_record("get sample");

                let image = match component_kind {
                    ComponentKind::F32 => image,
                    ComponentKind::F64 => image.to_kind(Kind::Float),
                    ComponentKind::U8 => image.to_kind(Kind::Float).g_div1(255.0),
                };
                debug_assert_eq!(
                    image.size3().unwrap(),
                    (channels as i64, height as i64, width as i64)
                );

                timing.set_record("change tensor kind");

                let bboxes: Vec<_> = bboxes
                    .into_iter()
                    .map(|entry| -> Result<_> {
                        let BBoxEntry {
                            class,
                            tlbr: [t, l, b, r],
                        } = *entry;
                        Ok(LabeledRatioBBox {
                            bbox: RatioBBox::try_from_tlbr([
                                t.try_into()?,
                                l.try_into()?,
                                b.try_into()?,
                                r.try_into()?,
                            ])?,
                            category_id: class as usize,
                        })
                    })
                    .try_collect()?;

                timing.set_record("compute bboxes");

                Ok((image, bboxes, timing))
            })
            .await?;

            timing.report();

            Fallible::Ok(DataRecord { image, bboxes })
        })
    }
}
