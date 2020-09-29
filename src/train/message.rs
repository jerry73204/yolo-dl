use crate::common::*;

#[derive(Debug, TensorLike)]
pub enum LoggingMessage {
    Images {
        #[tensor_like(clone)]
        tag: String,
        images: Vec<Tensor>,
    },
    ImageWithBBox {
        #[tensor_like(clone)]
        tag: String,
        image: Tensor,
        #[tensor_like(clone)]
        bboxes: Vec<LabeledRatioBBox>,
    },
}

impl LoggingMessage {
    pub fn new_images<S, T>(tag: S, image: &[T]) -> Self
    where
        S: ToString,
        T: Borrow<Tensor>,
    {
        let tag = tag.to_string();
        let images: Vec<_> = image
            .iter()
            .map(|tensor| tensor.borrow().shallow_clone())
            .collect();

        Self::Images { tag, images }
    }

    pub fn new_image_with_bboxes<S, T>(tag: S, image: T, bboxes: &[LabeledRatioBBox]) -> Self
    where
        S: ToString,
        T: Borrow<Tensor>,
    {
        let tag = tag.to_string();
        let image = image.borrow().shallow_clone();
        let bboxes = bboxes.to_owned();

        Self::ImageWithBBox { tag, image, bboxes }
    }
}

impl Clone for LoggingMessage {
    fn clone(&self) -> Self {
        self.shallow_clone()
    }
}
