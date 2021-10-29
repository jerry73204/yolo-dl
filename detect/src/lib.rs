mod common;
pub mod config;
pub mod input_stream;

use crate::{
    common::*,
    config::{Config, OutputConfig},
    input_stream::{InputRecord, InputStream},
};
use tch_goodies::MergedDenseDetection;

pub async fn start(config: Arc<Config>) -> Result<()> {
    let num_devices = config.model.devices.len();
    let (output_tx, output_rx) = async_channel::bounded(num_cpus::get() * 2);

    // load model
    let workers: Vec<_> = stream::iter(config.model.devices.clone())
        .par_map_init_unordered(
            num_devices,
            || config.clone(),
            move |config, device| {
                move || -> Result<_> {
                    let vs = nn::VarStore::new(device);
                    let root = vs.root();
                    let graph = model_graph::Graph::load_newslab_v1_json(&config.model.cfg_file)?;
                    let model = YoloModel::from_graph(root, &graph)?;

                    Ok((vs, model))
                }
            },
        )
        .try_collect()
        .await?;

    // load dataset
    let input_stream = InputStream::new(config.clone()).await?;

    // scatter input to workers
    let (scatter_fut, data_rx) = input_stream.stream()?.par_scatter(None);

    let inference_futs = {
        workers.into_iter().map(|(vs, mut model)| {
            let data_rx = data_rx.clone();
            let config = config.clone();
            let device = vs.device();
            let output_tx = output_tx.clone();
            let OutputConfig {
                nms_iou_thresh,
                nms_conf_thresh,
                ..
            } = config.output;
            let mut yolo_inference = YoloInferenceInit {
                nms_iou_thresh,
                nms_conf_thresh,
                suppress_by_class: false,
            }
            .build()
            .unwrap();

            async move {
                while let Ok(record) = data_rx.recv().await {
                    let (model_, yolo_inference_, images, inferences) =
                        tokio::task::spawn_blocking(move || -> Result<_> {
                            let record = record?;
                            let images = record.images.to_device(device);
                            let output = model.forward_t(&images, false)?;
                            let output = MergedDenseDetection::try_from(output)?;
                            let inferences = yolo_inference.forward(&output).to_device(Device::Cpu);
                            Ok((model, yolo_inference, record, inferences))
                        })
                        .await??;

                    model = model_;
                    yolo_inference = yolo_inference_;
                    output_tx.send((images, inferences)).await.unwrap();
                }
                Ok(())
            }
        })
    };

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
                    let pred_bboxes: Vec<RatioCyCxHW<R64>> = inference.bbox.try_into()?;

                    let image_size: PixelSize<R64> = {
                        let (_c, image_h, image_w) = image.size3().unwrap();
                        PixelSize::from_hw(image_h, image_w)
                            .unwrap()
                            .cast()
                            .unwrap()
                    };

                    let mut canvas: Mat =
                        TensorAsImage::new(image, ShapeConvention::Chw)?.try_into_cv()?;

                    // plot target boxes
                    target_bboxes.iter().try_for_each(|bbox| -> Result<_> {
                        let rect: PixelCyCxHW<i32> =
                            bbox.rect.to_pixel_cycxhw(&image_size).cast().unwrap();

                        imgproc::rectangle(
                            &mut canvas,
                            rect.into(),
                            Scalar::new(255.0, 255.0, 0.0, 0.0), // color
                            1,                                   // thickness
                            imgproc::LINE_8,                     // line_type
                            0,                                   // shift
                        )?;

                        Ok(())
                    })?;

                    // plot predicted boxes
                    pred_bboxes.iter().try_for_each(|cycxhw| -> Result<_> {
                        let rect: PixelCyCxHW<i32> =
                            cycxhw.to_pixel_cycxhw(&image_size).cast().unwrap();

                        imgproc::rectangle(
                            &mut canvas,
                            rect.into(),
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
        scatter_fut.map(|_| Fallible::Ok(())),
        futures::future::try_join_all(inference_futs),
        output_fut.map(|_| Fallible::Ok(())),
    )?;

    Ok(())
}
