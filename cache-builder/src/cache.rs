use crate::{common::*, utils};
use async_std::{
    fs::{File, OpenOptions},
    io::{BufWriter, SeekFrom},
    path::Path,
};
use futures::io::Cursor;

pub const MAGIC: [u8; 8] = [b'a', b'e', b'o', b'n', b'd', b'a', b't', b'a'];

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

pub struct DatasetWriterInit<I> {
    pub num_images: usize,
    pub shape: [u32; 3],
    pub alignment: Option<usize>,
    pub classes: IndexMap<u32, String>,
    pub images: I,
}

impl<I> DatasetWriterInit<I> {
    pub async fn write<P, B, D, C, E>(self, output_path: P) -> Result<()>
    where
        P: AsRef<Path>,
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
        let alignment = alignment.unwrap_or(64);
        let component_kind = C::KIND;
        let component_size = C::SIZE;

        // initialize header
        let mut header = Header {
            magic: MAGIC,
            component_kind,
            shape,
            alignment: alignment as u32,
            data_offset: 0,
            bbox_offset: 0,
        };
        let class_entries: Vec<_> = classes
            .into_iter()
            .map(|(index, name)| ClassEntry { index, name })
            .collect();
        let mut image_entries = vec![ImageEntry::zero(); num_images as usize];

        // compute offsets
        let header_size = bincode::serialized_size(&header)? as usize;
        let class_entries_size = bincode::serialized_size(&class_entries)? as usize;
        let image_entries_size = bincode::serialized_size(&image_entries)? as usize;
        let per_image_size = {
            let [c, h, w] = shape;
            (component_size as u32 * c * h * w) as usize
        };
        let per_data_size = { utils::nearest_multiple(per_image_size, alignment) };
        let data_size = per_data_size * num_images as usize;

        let header_offset = 0usize;
        let class_entries_offset = header_offset + header_size;
        let image_entries_offset = class_entries_offset + class_entries_size;
        let data_offset =
            utils::nearest_multiple(image_entries_offset + image_entries_size, alignment);
        let bbox_offset = utils::nearest_multiple(data_offset + data_size, alignment);

        // finish header
        let header = {
            header.data_offset = data_offset as u64;
            header.bbox_offset = bbox_offset as u64;
            header
        };

        // set file size
        {
            let output_file = File::create(output_path).await?;
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
                            assert!(end <= bbox_offset);
                            let slice = &mmap.as_ref()[begin..end];
                            // to convert from an immutable to mutable slice
                            // we have to bypass ownership checking here
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

        {
            let file = OpenOptions::new().write(true).open(output_path).await?;
            file.set_len(bbox_offset as u64).await?;
            let mut writer = BufWriter::new(file);
            writer.seek(SeekFrom::Start(bbox_offset as u64)).await?;
            writer
                .write_all(&bincode::serialize(&bbox_entries)?)
                .await?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DataIndex {
    data_range: Range<usize>,
    bbox_range: Range<usize>,
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub header: Header,
    pub classes: IndexMap<usize, String>,
    pub image_entries: Vec<ImageEntry>,
    pub bbox_entries: Vec<BBoxEntry>,
    pub per_image_size: usize,
    pub per_data_size: usize,
    data_indexes: Vec<DataIndex>,
    mmap: Arc<Mmap>,
}

impl Dataset {
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        // create memory mapped file
        let mmap = unsafe {
            let output_file = std::fs::File::open(path)?;
            let mmap = MmapOptions::new().map(&output_file)?;
            mmap
        };
        let mmap_slice = mmap.as_ref();
        let mut cursor = std::io::Cursor::new(mmap_slice);

        #[derive(Debug, Deserialize)]
        struct Prefix {
            header: Header,
            class_entries: Vec<ClassEntry>,
            image_entries: Vec<ImageEntry>,
        }

        let Prefix {
            header,
            class_entries,
            image_entries,
        } = bincode::deserialize_from(&mut cursor)?;

        // deserialize header
        let Header {
            magic,
            alignment,
            shape,
            component_kind,
            data_offset,
            bbox_offset,
        } = header;
        let component_size = component_kind.component_size();

        // sanity check
        ensure!(magic == MAGIC, "file magic does not match");
        ensure!(
            bbox_offset >= data_offset,
            "assert data_offset ({}) <= bbox_offset ({}) but failed",
            data_offset,
            bbox_offset
        );

        let num_classes = class_entries.len();
        let classes: IndexMap<_, _> = class_entries
            .into_iter()
            .map(|ClassEntry { index, name }| (index as usize, name))
            .collect();
        ensure!(classes.len() == num_classes, "duplicated class id found");

        // calculate data and bbox section offsets
        let per_image_size = {
            let [c, h, w] = shape;
            component_size * c as usize * h as usize * w as usize
        };
        let per_data_size = utils::nearest_multiple(per_image_size, alignment as usize);

        let diff = (bbox_offset - data_offset) as usize;
        ensure!(
            (diff % per_data_size == 0) && (diff / per_data_size == image_entries.len()),
            "the size of data section does not match the number of images in header"
        );

        // deserialize bboxes
        let bbox_ranges = image_entries.iter().scan(0usize, |bbox_index, entry| {
            let ImageEntry { num_bboxes, .. } = *entry;
            let begin = *bbox_index;
            let end = begin + num_bboxes as usize;
            *bbox_index = end;
            Some(begin..end)
        });

        let num_bboxes = bbox_ranges.last().map(|range| range.end).unwrap_or(0);

        cursor.set_position(bbox_offset as u64);
        let bbox_entries: Vec<_> = (0..num_bboxes)
            .map(|_| -> Result<_> {
                let bbox_entry: BBoxEntry = bincode::deserialize_from(&mut cursor)?;
                Ok(bbox_entry)
            })
            .try_collect()?;

        // compute bbox ranges
        let bbox_indexes_iter = image_entries
            .iter()
            .map(|entry| entry.num_bboxes as usize)
            .scan(0, |bbox_index, num_bboxes| {
                let begin = *bbox_index;
                let end = begin + num_bboxes;
                *bbox_index = end;
                Some((begin, end))
            });

        // build data slices per image
        let data_indexes: Vec<_> = (data_offset..bbox_offset)
            .step_by(per_data_size)
            .zip_eq(bbox_indexes_iter)
            .map(move |(offset, (bbox_begin, bbox_end))| {
                let data_begin = offset as usize;
                let data_end = data_begin + per_image_size;
                DataIndex {
                    data_range: data_begin..data_end,
                    bbox_range: bbox_begin..bbox_end,
                }
            })
            .collect();

        Ok(Self {
            header,
            classes,
            image_entries,
            bbox_entries,
            per_image_size,
            per_data_size,
            data_indexes,
            mmap: Arc::new(mmap),
        })
    }

    pub fn image_iter(&self) -> impl Iterator<Item = (&[u8], &[BBoxEntry])> {
        let Self {
            bbox_entries,
            data_indexes,
            mmap,
            ..
        } = self;

        data_indexes.iter().map(move |data_index| {
            let DataIndex {
                data_range,
                bbox_range,
            } = data_index.clone();
            let data = &mmap.as_ref()[data_range];
            let bboxes = &bbox_entries[bbox_range];
            (data, bboxes)
        })
    }
}

pub struct ImageItem<B, D> {
    pub bboxes: B,
    pub data: D,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    pub magic: [u8; 8],
    pub component_kind: ComponentKind,
    pub shape: [u32; 3],
    pub alignment: u32,
    pub data_offset: u64,
    pub bbox_offset: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassEntry {
    pub index: u32,
    #[serde(with = "serde_fixed_length_string")]
    pub name: String,
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ComponentKind {
    F32 = 0,
    F64 = 1,
    U8 = 2,
}

impl ComponentKind {
    pub fn component_size(&self) -> usize {
        match self {
            Self::F32 => mem::size_of::<f32>(),
            Self::F64 => mem::size_of::<f64>(),
            Self::U8 => mem::size_of::<u8>(),
        }
    }
}

mod serde_fixed_length_string {
    use super::*;

    serde_big_array::big_array! {
        BigArray; SIZE
    }

    const SIZE: usize = 64;

    pub fn serialize<S>(from: &String, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes = cesu8::to_java_cesu8(from.as_str());
        if bytes.len() > SIZE {
            return Err(S::Error::custom(format!(
                "the size of Java CESU-8 encoding of string exceed the maximum size {}",
                SIZE
            )));
        }
        let array = {
            let mut array = [0u8; SIZE];
            array[0..(bytes.len())].copy_from_slice(bytes.borrow());
            array
        };
        array.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<String, D::Error>
    where
        D: Deserializer<'de>,
    {
        let array = <[u8; SIZE]>::deserialize(deserializer)?;
        let nul_pos = array
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or_else(|| array.len());
        let text = cesu8::from_java_cesu8(&array[0..nul_pos])
            .map_err(|err| {
                D::Error::custom(format!("failed to decode as CESU-8 string: {:?}", err))
            })?
            .to_string();
        Ok(text)
    }
}
