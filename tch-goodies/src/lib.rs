use anyhow::{bail, Error, Result};
use image::{DynamicImage, GenericImageView, ImageBuffer, Pixel};
use itertools::Itertools;
use std::ops::Deref;
use tch::{kind::Element, vision, Kind, Tensor};

pub trait TensorExt {
    fn resize2d(&self, new_height: i64, new_width: i64) -> Result<Tensor>;
    fn resize2d_exact(&self, new_height: i64, new_width: i64) -> Result<Tensor>;
    fn resize2d_letterbox(&self, new_height: i64, new_width: i64) -> Result<Tensor>;
    fn swish(&self) -> Tensor;
    fn hard_swish(&self) -> Tensor;
    fn mish(&self) -> Tensor;
    fn hard_mish(&self) -> Tensor;
    // fn normalize_channels(&self) -> Tensor;
    // fn normalize_channels_softmax(&self) -> Tensor;
}

impl TensorExt for Tensor {
    fn resize2d(&self, new_height: i64, new_width: i64) -> Result<Tensor> {
        tch::no_grad(|| match (self.kind(), self.size().as_slice()) {
            (Kind::Float, &[_n_channels, _height, _width]) => {
                let resized = vision::image::resize_preserve_aspect_ratio(
                    &(self * 255.0).to_kind(Kind::Uint8),
                    new_width,
                    new_height,
                )?
                .to_kind(Kind::Float)
                    / 255.0;
                Ok(resized)
            }
            (Kind::Uint8, &[_n_channels, _height, _width]) => {
                let resized =
                    vision::image::resize_preserve_aspect_ratio(self, new_width, new_height)?;
                Ok(resized)
            }
            (_, &[_n_channels, _height, _width]) => bail!("unsupported data kind"),
            (Kind::Float, &[batch_size, _n_channels, _height, _width]) => {
                let self_scaled = (self * 255.0).to_kind(Kind::Uint8);
                let resized_vec: Vec<_> = (0..batch_size)
                    .map(|index| -> Result<_> {
                        let resized = vision::image::resize_preserve_aspect_ratio(
                            &self_scaled.select(0, index),
                            new_width,
                            new_height,
                        )?;
                        Ok(resized)
                    })
                    .try_collect()?;
                let resized = Tensor::stack(resized_vec.as_slice(), 0);
                let resized_scaled = resized.to_kind(Kind::Float) / 255.0;
                Ok(resized_scaled)
            }
            (Kind::Uint8, &[batch_size, _n_channels, _height, _width]) => {
                let resized_vec: Vec<_> = (0..batch_size)
                    .map(|index| -> Result<_> {
                        let resized = vision::image::resize_preserve_aspect_ratio(
                            &self.select(0, index),
                            new_width,
                            new_height,
                        )?;
                        Ok(resized)
                    })
                    .try_collect()?;
                let resized = Tensor::stack(resized_vec.as_slice(), 0);
                Ok(resized)
            }
            (_, &[_batch_size, _n_channels, _height, _width]) => bail!("unsupported data kind"),
            _ => bail!("invalid shape: expect three or four dimensions"),
        })
    }

    fn resize2d_exact(&self, new_height: i64, new_width: i64) -> Result<Tensor> {
        tch::no_grad(|| match (self.kind(), self.size().as_slice()) {
            (Kind::Uint8, &[_n_channels, _height, _width]) => {
                let resized = vision::image::resize(self, new_width, new_height)?;
                Ok(resized)
            }
            (Kind::Float, &[_n_channels, _height, _width]) => {
                let resized = vision::image::resize(
                    &(self * 255.0).to_kind(Kind::Uint8),
                    new_width,
                    new_height,
                )?
                .to_kind(Kind::Float)
                    / 255.0;
                Ok(resized)
            }
            (_, &[_n_channels, _height, _width]) => bail!("unsupported data kind"),
            (Kind::Float, &[batch_size, _n_channels, _height, _width]) => {
                let self_scaled = (self * 255.0).to_kind(Kind::Uint8);
                let resized_vec: Vec<_> = (0..batch_size)
                    .map(|index| -> Result<_> {
                        let resized = vision::image::resize(
                            &self_scaled.select(0, index),
                            new_width,
                            new_height,
                        )?;
                        Ok(resized)
                    })
                    .try_collect()?;
                let resized = Tensor::stack(resized_vec.as_slice(), 0);
                let resized_scaled = resized.to_kind(Kind::Float) / 255.0;
                Ok(resized_scaled)
            }
            (Kind::Uint8, &[batch_size, _n_channels, _height, _width]) => {
                let resized_vec: Vec<_> = (0..batch_size)
                    .map(|index| -> Result<_> {
                        let resized =
                            vision::image::resize(&self.select(0, index), new_width, new_height)?;
                        Ok(resized)
                    })
                    .try_collect()?;
                let resized = Tensor::stack(resized_vec.as_slice(), 0);
                Ok(resized)
            }
            (_, &[_batch_size, _n_channels, _height, _width]) => bail!("unsupported data kind"),
            _ => bail!("invalid shape: expect three or four dimensions"),
        })
    }

    fn resize2d_letterbox(&self, new_height: i64, new_width: i64) -> Result<Tensor> {
        let inner_rect = |height: i64, width: i64| {
            let scale_h = new_height as f64 / height as f64;
            let scale_w = new_width as f64 / width as f64;
            let (inner_h, inner_w) = if scale_h <= scale_w {
                (
                    new_height,
                    (width as f64 * new_height as f64 / height as f64) as i64,
                )
            } else {
                (
                    (height as f64 * new_width as f64 / width as f64) as i64,
                    new_width,
                )
            };
            let (top, left) = ((new_height - inner_h) / 2, (new_width - inner_w) / 2);
            (inner_h, inner_w, top, left)
        };

        tch::no_grad(|| match (self.kind(), self.size().as_slice()) {
            (Kind::Uint8, &[channels, height, width]) => {
                let (inner_h, inner_w, top, left) = inner_rect(height, width);
                let inner = vision::image::resize(self, inner_w, inner_h)?;
                let outer = Tensor::zeros(
                    &[channels, new_height, new_width],
                    (self.kind(), self.device()),
                );
                outer
                    .narrow(1, top, inner_h)
                    .narrow(2, left, inner_w)
                    .copy_(&inner);
                Ok(outer)
            }
            (Kind::Float, &[channels, height, width]) => {
                let (inner_h, inner_w, top, left) = inner_rect(height, width);
                let inner =
                    vision::image::resize(&(self * 255.0).to_kind(Kind::Uint8), inner_w, inner_h)?
                        .to_kind(Kind::Float)
                        / 255.0;
                let outer = Tensor::zeros(
                    &[channels, new_height, new_width],
                    (self.kind(), self.device()),
                );
                outer
                    .narrow(1, top, inner_h)
                    .narrow(2, left, inner_w)
                    .copy_(&inner);
                Ok(outer)
            }
            (_, &[_n_channels, _height, _width]) => bail!("unsupported data kind"),
            (Kind::Float, &[batch_size, channels, height, width]) => {
                let (inner_h, inner_w, top, left) = inner_rect(height, width);
                let scaled = (self * 255.0).to_kind(Kind::Uint8);
                let resized_vec: Vec<_> = (0..batch_size)
                    .map(|index| vision::image::resize(&scaled.select(0, index), inner_w, inner_h))
                    .try_collect()?;
                let inner = Tensor::stack(resized_vec.as_slice(), 0).to_kind(Kind::Float) / 255.0;
                let outer = Tensor::zeros(
                    &[batch_size, channels, new_height, new_width],
                    (self.kind(), self.device()),
                );
                outer
                    .narrow(2, top, inner_h)
                    .narrow(3, left, inner_w)
                    .copy_(&inner);
                Ok(outer)
            }
            (Kind::Uint8, &[batch_size, channels, height, width]) => {
                let (inner_h, inner_w, top, left) = inner_rect(height, width);
                let resized_vec: Vec<_> = (0..batch_size)
                    .map(|index| -> Result<_> {
                        let resized =
                            vision::image::resize(&self.select(0, index), inner_w, inner_h)?;
                        Ok(resized)
                    })
                    .try_collect()?;
                let inner = Tensor::stack(resized_vec.as_slice(), 0);
                let outer = Tensor::zeros(
                    &[batch_size, channels, new_height, new_width],
                    (self.kind(), self.device()),
                );
                outer
                    .narrow(2, top, inner_h)
                    .narrow(3, left, inner_w)
                    .copy_(&inner);
                Ok(outer)
            }
            (_, &[_batch_size, _n_channels, _height, _width]) => bail!("unsupported data kind"),
            _ => bail!("invalid shape: expect three or four dimensions"),
        })
    }

    fn swish(&self) -> Tensor {
        self * self.sigmoid()
    }

    fn hard_swish(&self) -> Tensor {
        self * (self + 3.0).clamp(0.0, 6.0) / 6.0
    }

    fn mish(&self) -> Tensor {
        self * &self.softplus().tanh()
    }

    fn hard_mish(&self) -> Tensor {
        let case1 = self.clamp(-2.0, 0.0);
        let case2 = self.clamp_min(0.0);
        (case1.pow(2.0) / 2.0 + &case1) + case2
    }

    // fn normalize_channels(&self) -> Tensor {
    //     todo!();
    // }

    // fn normalize_channels_softmax(&self) -> Tensor {
    //     todo!();
    // }
}

pub trait IntoTensor {
    fn into_tensor(self) -> Tensor;
}

impl<P, Container> IntoTensor for &ImageBuffer<P, Container>
where
    P: Pixel + 'static,
    P::Subpixel: 'static + Element,
    Container: Deref<Target = [P::Subpixel]>,
{
    fn into_tensor(self) -> Tensor {
        let (width, height) = self.dimensions();
        let height = height as usize;
        let width = width as usize;
        let channels = P::CHANNEL_COUNT as usize;

        let buffer = unsafe {
            let buf_len = channels * height * width;
            let mut buffer: Vec<P::Subpixel> = Vec::with_capacity(buf_len);
            let ptr = buffer.as_mut_ptr();
            self.enumerate_pixels().for_each(|(x, y, pixel)| {
                let x = x as usize;
                let y = y as usize;
                pixel
                    .channels()
                    .iter()
                    .cloned()
                    .enumerate()
                    .for_each(|(c, component)| {
                        *ptr.add(x + width * (y + height * c)) = component;
                    });
            });
            buffer.set_len(buf_len);
            buffer
        };

        Tensor::of_slice(&buffer).view([channels as i64, height as i64, width as i64])
    }
}

pub trait TryIntoTensor {
    type Error;

    fn try_into_tensor(self) -> Result<Tensor, Self::Error>;
}

impl TryIntoTensor for &DynamicImage {
    type Error = Error;

    fn try_into_tensor(self) -> Result<Tensor, Self::Error> {
        let tensor = match self {
            DynamicImage::ImageLuma8(image) => image.into_tensor(),
            DynamicImage::ImageLumaA8(image) => image.into_tensor(),
            DynamicImage::ImageRgb8(image) => image.into_tensor(),
            DynamicImage::ImageRgba8(image) => image.into_tensor(),
            DynamicImage::ImageBgr8(image) => image.into_tensor(),
            DynamicImage::ImageBgra8(image) => image.into_tensor(),
            _ => bail!("cannot convert an image with u16 components to a tensor"),
        };
        Ok(tensor)
    }
}
