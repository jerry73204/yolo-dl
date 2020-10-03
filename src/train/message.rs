use crate::{common::*, util::CowTensor};

#[derive(Debug, Clone)]
pub struct LoggingMessage {
    pub tag: Cow<'static, str>,
    pub kind: LoggingMessageKind,
}

#[derive(Debug)]
pub enum LoggingMessageKind {
    TrainingStep {
        step: usize,
        loss: f32,
    },
    TrainingOutput {
        input: Tensor,
        output: YoloOutput,
        losses: YoloLossOutput,
    },
    Images {
        images: Vec<Tensor>,
    },
    ImagesWithBBoxes {
        tuples: Vec<(Tensor, Vec<LabeledRatioBBox>)>,
    },
}

impl LoggingMessage {
    pub fn new_training_step<S>(tag: S, step: usize, loss: f32) -> Self
    where
        S: Into<Cow<'static, str>>,
    {
        Self {
            tag: tag.into(),
            kind: LoggingMessageKind::TrainingStep { step, loss },
        }
    }

    pub fn new_training_output<S>(
        tag: S,
        input: &Tensor,
        output: &YoloOutput,
        losses: &YoloLossOutput,
    ) -> Self
    where
        S: Into<Cow<'static, str>>,
    {
        Self {
            tag: tag.into(),
            kind: LoggingMessageKind::TrainingOutput {
                input: input.shallow_clone(),
                output: output.shallow_clone(),
                losses: losses.shallow_clone(),
            },
        }
    }

    pub fn new_images<'a, S, I, T>(tag: S, images: I) -> Self
    where
        S: Into<Cow<'static, str>>,
        I: IntoIterator<Item = T>,
        T: Into<CowTensor<'a>>,
    {
        Self {
            tag: tag.into(),
            kind: LoggingMessageKind::Images {
                images: images
                    .into_iter()
                    .map(|tensor| tensor.into().into_owned())
                    .collect_vec(),
            },
        }
    }

    pub fn new_images_with_bboxes<'a, S, I, IB, B, T>(tag: S, tuples: I) -> Self
    where
        S: Into<Cow<'static, str>>,
        I: IntoIterator<Item = (T, IB)>,
        IB: IntoIterator<Item = B>,
        B: Borrow<LabeledRatioBBox>,
        T: Into<CowTensor<'a>>,
    {
        Self {
            tag: tag.into(),
            kind: LoggingMessageKind::ImagesWithBBoxes {
                tuples: tuples
                    .into_iter()
                    .map(|(tensor, bboxes)| {
                        (
                            tensor.into().into_owned(),
                            bboxes
                                .into_iter()
                                .map(|bbox| bbox.borrow().to_owned())
                                .collect_vec(),
                        )
                    })
                    .collect_vec(),
            },
        }
    }
}

impl Clone for LoggingMessageKind {
    fn clone(&self) -> Self {
        match *self {
            Self::TrainingStep { step, loss } => Self::TrainingStep { step, loss },
            Self::TrainingOutput {
                ref input,
                ref output,
                ref losses,
            } => Self::TrainingOutput {
                input: input.shallow_clone(),
                output: output.shallow_clone(),
                losses: losses.shallow_clone(),
            },
            Self::Images { ref images } => Self::Images {
                images: images
                    .iter()
                    .map(|image| image.shallow_clone())
                    .collect_vec(),
            },
            Self::ImagesWithBBoxes { ref tuples } => Self::ImagesWithBBoxes {
                tuples: tuples
                    .iter()
                    .map(|(image, bboxes)| (image.shallow_clone(), bboxes.clone()))
                    .collect_vec(),
            },
        }
    }
}
