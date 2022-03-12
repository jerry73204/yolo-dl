mod common;
mod config;
mod input_stream;

use crate::{
    common::*,
    config::{Config, OutputConfig},
    input_stream::{InputRecord, InputStream},
};
use bbox::{prelude::*, CyCxHW, Transform, HW};
use itertools::izip;
use opencv::{
    core::{Mat, Scalar, Vector},
    imgcodecs, imgproc,
};
use std::{path::PathBuf, sync::Arc};
use structopt::StructOpt;
use tch::{nn, Device, IndexOp};
use tch_goodies::{CyCxHWTensor, CyCxHWTensorUnchecked, MergedDenseDetection, Pixel, Ratio};
use tch_tensor_like::TensorLike;
use yolo_dl::{loss::YoloInferenceInit, model::YoloModel};

#[derive(Debug, Clone, StructOpt)]
/// Train YOLO model
struct Opts {
    #[structopt(long, default_value = "detect.json5")]
    /// configuration file
    pub config_file: PathBuf,
}

#[tokio::main]
pub async fn main() -> Result<()> {
    // parse arguments
    let Opts { config_file } = Opts::from_args();
    let config = Arc::new(
        Config::open(&config_file)
            .with_context(|| format!("failed to load config file '{}'", config_file.display()))?,
    );

    let num_devices = config.model.devices.len();
    let (output_tx, output_rx) = async_channel::bounded(num_cpus::get() * 2);

    // load model
    let workers: Vec<_> = stream::iter(config.model.devices.clone())
        .par_map_unordered(num_devices, {
            let config = config.clone();
            move |device| {
                let config = config.clone();

                move || -> Result<_> {
                    let vs = nn::VarStore::new(device);
                    let root = vs.root();
                    let graph = model_graph::Graph::load_newslab_v1_json(&config.model.cfg_file)?;
                    let model = YoloModel::from_graph(root, &graph)?;

                    Ok((vs, model))
                }
            }
        })
        .try_collect()
        .await?;

    // load dataset
    let input_stream = InputStream::new(config.clone()).await?;

    // scatter input to workers
    let data_stream = input_stream.stream()?.shared();

    let inference_futs = workers.into_iter().map({
        let config = config.clone();
        move |(vs, model)| {
            let data_stream = data_stream.clone();
            let config = config.clone();
            let device = vs.device();
            let output_tx = output_tx.clone();
            let OutputConfig {
                nms_iou_thresh,
                nms_conf_thresh,
                ..
            } = config.output;
            let yolo_inference = YoloInferenceInit {
                nms_iou_thresh,
                nms_conf_thresh,
                suppress_by_class: false,
            }
            .build()
            .unwrap();

            async move {
                data_stream
                    .try_fold(
                        (model, yolo_inference),
                        |(mut model, yolo_inference), record| async {
                            let images = record.images.to_device(device);
                            let output = model.forward_t(&images, false)?;
                            let output = MergedDenseDetection::try_from(output)?;
                            let inferences = yolo_inference.forward(&output).to_device(Device::Cpu);
                            output_tx.send((record, inferences)).await.unwrap();
                            Ok((model, yolo_inference))
                        },
                    )
                    .await?;
                Ok(())
            }
        }
    });

    let output_fut = {
        let output_dir = Arc::new(config.output.output_dir.clone());
        fs::create_dir_all(&*output_dir)?;

        output_rx
            .flat_map(|(record, inferences)| {
                let InputRecord {
                    indexes,
                    images,
                    bboxes,
                } = record;
                let minibatch_size = images.size4().unwrap().0;

                stream::iter((0..minibatch_size).zip_eq(indexes).zip_eq(bboxes).map(
                    move |((minibatch_index, index), target_bboxes)| {
                        Ok((
                            index,
                            minibatch_index,
                            images.shallow_clone(),
                            target_bboxes,
                            inferences.shallow_clone(),
                        ))
                    },
                ))
            })
            .try_par_for_each_blocking(None, move |args| {
                let output_dir = output_dir.clone();

                move || -> Result<_> {
                    let (index, minibatch_index, images, target_bboxes, inferences) = args;

                    let image = images.i((minibatch_index, .., .., ..));
                    let inference = inferences.batch_select(minibatch_index);
                    let pred_bboxes: Vec<Ratio<CyCxHW<R64>>> = {
                        let cycxhw: CyCxHWTensor = (&inference.bbox).into();
                        let CyCxHWTensorUnchecked { cy, cx, h, w } = cycxhw.into();
                        let cy: Vec<f32> = cy.into();
                        let cx: Vec<f32> = cx.into();
                        let h: Vec<f32> = h.into();
                        let w: Vec<f32> = w.into();

                        izip!(cy, cx, h, w)
                            .map(|(cy, cx, h, w)| {
                                Ratio(CyCxHW::from_cycxhw([cy, cx, h, w]).cast::<R64>())
                            })
                            .collect()
                    };

                    let image_size: Pixel<HW<R64>> = {
                        let (_c, image_h, image_w) = image.size3().unwrap();

                        Pixel(HW::from_hw([image_h, image_w]).cast())
                    };

                    let mut canvas: Mat =
                        TensorAsImage::new(image, ShapeConvention::Chw)?.try_into_cv()?;

                    let transform =
                        Transform::from_sizes_letterbox(HW::unit(), image_size.0.cast::<R64>());

                    // plot target boxes
                    target_bboxes.iter().try_for_each(|bbox| -> Result<_> {
                        let rect: Pixel<CyCxHW<i32>> =
                            Pixel((&transform * &bbox.rect).cast::<i32>());

                        imgproc::rectangle(
                            &mut canvas,
                            rect.0.into(),
                            Scalar::new(255.0, 255.0, 0.0, 0.0), // color
                            1,                                   // thickness
                            imgproc::LINE_8,                     // line_type
                            0,                                   // shift
                        )?;

                        Ok(())
                    })?;

                    // plot predicted boxes
                    pred_bboxes.iter().try_for_each(|cycxhw| -> Result<_> {
                        let rect: Pixel<CyCxHW<i32>> =
                            Pixel((&transform * &cycxhw.0).cast::<i32>());

                        imgproc::rectangle(
                            &mut canvas,
                            rect.0.into(),
                            Scalar::new(0.0, 255.0, 255.0, 0.0), // color
                            1,                                   // thickness
                            imgproc::LINE_8,                     // line_type
                            0,                                   // shift
                        )?;

                        Ok(())
                    })?;

                    // save image
                    let output_file = output_dir.join(format!("{}.jpg", index));
                    imgcodecs::imwrite(output_file.to_str().unwrap(), &canvas, &Vector::new())?;

                    Ok(())
                    // TODO
                }
            })
    };

    futures::try_join!(
        futures::future::try_join_all(inference_futs),
        output_fut.map(|_| anyhow::Ok(())),
    )?;

    Ok(())
}
