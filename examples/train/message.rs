use crate::common::*;

#[derive(Debug, TensorLike)]
pub enum LoggingMessage {
    Images {
        #[tensor_like(clone)]
        tag: String,
        images: Vec<Tensor>,
    },
}

impl Clone for LoggingMessage {
    fn clone(&self) -> Self {
        self.shallow_clone()
    }
}
