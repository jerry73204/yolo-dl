use crate::common::*;
use typenum::consts::*;

pub const MAGIC: [u8; 8] = [b'a', b'e', b'o', b'n', b'd', b'a', b't', b'a'];

fn nearest_multiple(value: usize, multiple: usize) -> usize {
    (value + multiple - 1) & !(multiple - 1)
}

pub trait Component
where
    Self: TriviallyTransmutable + Copy + Sized + Sync + Send,
{
    const KIND: ComponentKind;
    const SIZE: usize = mem::size_of::<Self>();
}

impl Component for u8 {
    const KIND: ComponentKind = ComponentKind::U8;
}

impl Component for f32 {
    const KIND: ComponentKind = ComponentKind::F32;
}

impl Component for f64 {
    const KIND: ComponentKind = ComponentKind::F64;
}

pub struct DatasetInit<I> {
    pub num_images: usize,
    pub shape: [u32; 3],
    pub alignment: Option<usize>,
    pub classes: IndexMap<u32, String>,
    pub images: I,
}

impl<I> DatasetInit<I> {
    pub async fn write<P, B, D, C, E>(self, output_path: P) -> Result<()>
    where
        P: AsRef<async_std::path::Path>,
        I: 'static
            + TryStream<Ok = ImageItem<B, D>, Error = E>
            + TryStreamExt
            + StreamExt
            + Send
            + Unpin,
        B: 'static + IntoIterator<Item = BBoxEntry> + Send,
        D: 'static + AsRef<[C]> + Send,
        C: Component,
        E: 'static + Sync + Send + Into<Error>,
        B::IntoIter: Send,
    {
        let output_path = output_path.as_ref();
        let Self {
            num_images,
            shape,
            alignment,
            classes,
            images,
        } = self;
        let num_classes = classes.len();
        let alignment = alignment.unwrap_or(64);
        let component_kind = C::KIND;
        let component_size = C::SIZE;

        // initialize header
        let mut header = Header {
            magic: MAGIC,
            num_classes: num_classes as u32,
            num_images: num_images as u32,
            component_kind,
            shape,
            alignment: alignment as u32,
            data_offset: 0,
            bbox_offset: 0,
        };
        let class_entries: Vec<_> = classes
            .iter()
            .map(|(&index, name)| -> Result<_> {
                Ok(ClassEntry {
                    index,
                    name: ArrayString::try_from_str(name)?,
                })
            })
            .try_collect()?;
        let mut image_entries = vec![ImageEntry::zero(); num_images as usize];

        // compute offsets
        let header_size = bincode::serialized_size(&header)? as usize;
        let class_entries_size = bincode::serialized_size(&class_entries)? as usize;
        let image_entries_size = bincode::serialized_size(&image_entries)? as usize;
        let per_image_size = {
            let [c, h, w] = shape;
            (component_size as u32 * c * h * w) as usize
        };
        let per_data_size = { nearest_multiple(per_image_size, alignment) };
        let data_size = per_data_size * num_images as usize;

        let header_offset = 0usize;
        let class_entries_offset = header_offset + header_size;
        let image_entries_offset = class_entries_offset + class_entries_size;
        let data_offset = nearest_multiple(image_entries_offset + image_entries_size, alignment);
        let bbox_offset = nearest_multiple(data_offset + data_size, alignment);

        // finish header
        let header = {
            header.data_offset = data_offset as u64;
            header.bbox_offset = bbox_offset as u64;
            header
        };

        // set file size
        {
            let output_file = async_std::fs::File::create(output_path).await?;
            output_file.set_len(bbox_offset as u64).await?;
        }

        // create memory mapped file
        let mut mmap = unsafe {
            let output_file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(output_path)?;
            let mmap = MmapOptions::new()
                .offset(0)
                .len(bbox_offset)
                .map_mut(&output_file)?;
            mmap
        };

        // write header and class entries
        {
            let mut cursor = Cursor::new(mmap.as_mut());
            cursor.write_all(&bincode::serialize(&header)?).await?;
            cursor
                .write_all(&bincode::serialize(&class_entries)?)
                .await?;
        }

        // write image data in parallel
        let mmap = Arc::new(mmap);

        let mut bbox_chunks: Vec<_> = {
            let mmap = mmap.clone();
            images
                .map_err(|err| err.into())
                .try_overflowing_enumerate()
                .try_par_then(None, move |(index, item)| {
                    let mmap = mmap.clone();
                    async move {
                        ensure!(
                            index < num_images,
                            "the number of stream items must not exceed num_images"
                        );

                        let ImageItem { bboxes, data } = item;
                        let data = data.as_ref();
                        let bytes = safe_transmute::transmute_to_bytes(data);
                        ensure!(bytes.len() == per_image_size, "image size does not match");

                        let chunk = unsafe {
                            let begin = data_offset + index * per_data_size;
                            let end = begin + per_image_size;
                            debug_assert!(end <= bbox_offset);
                            let slice = &mmap.as_ref()[begin..end];
                            let chunk =
                                slice::from_raw_parts_mut(slice.as_ptr() as *mut _, per_image_size);
                            chunk
                        };
                        Cursor::new(chunk).write_all(bytes).await?;

                        Fallible::Ok((index, bboxes))
                    }
                })
                .try_collect()
                .await?
        };

        ensure!(
            bbox_chunks.len() == num_images,
            "the number of stream items must be equal to num_images"
        );

        let mut mmap = Arc::try_unwrap(mmap).unwrap();

        // build bbox entries and update image entries in the mean time
        bbox_chunks.sort_by_cached_key(|(image_index, _bboxes)| *image_index);

        let bbox_entries: Vec<_> = bbox_chunks
            .into_iter()
            .scan(0, |bbox_index, (image_index, bboxes)| {
                let bboxes: Vec<_> = bboxes.into_iter().collect();
                let num_bboxes = bboxes.len();
                *bbox_index += num_bboxes;
                image_entries[image_index].num_bboxes = num_bboxes as u32;
                Some(bboxes)
            })
            .flatten()
            .collect();

        // finish image entries
        let image_entries = image_entries;

        // write image entries
        Cursor::new(&mut mmap.as_mut()[image_entries_offset..bbox_offset])
            .write_all(&bincode::serialize(&image_entries)?)
            .await?;

        // write bbox entries
        drop(mmap);

        let mut writer =
            async_std::io::BufWriter::new(async_std::fs::File::open(output_path).await?);
        writer
            .write_all(&bincode::serialize(&bbox_entries)?)
            .await?;

        Ok(())
    }
}

pub struct ImageItem<B, D> {
    pub bboxes: B,
    pub data: D,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    magic: [u8; 8],
    pub num_images: u32,
    pub num_classes: u32,
    pub component_kind: ComponentKind,
    pub shape: [u32; 3],
    pub alignment: u32,
    pub data_offset: u64,
    pub bbox_offset: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassEntry {
    pub index: u32,
    pub name: ArrayString<U23>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEntry {
    pub bbox_index_begin: u32,
    pub num_bboxes: u32,
}

impl ImageEntry {
    pub(crate) fn zero() -> Self {
        Self {
            bbox_index_begin: 0,
            num_bboxes: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BBoxEntry {
    pub tlbr: [f64; 4],
    pub class: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComponentKind {
    F32 = 0,
    F64 = 1,
    U8 = 2,
}
