mod activation;
mod helper;
mod model;
mod module;

pub use activation::*;
pub use helper::*;
pub use model::*;
pub use module::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::*;

    #[test]
    fn yolo_v5_small_serde_test() -> Result<()> {
        let init = yolo_v5_small_init(3, 80);
        let text = serde_json::to_string_pretty(&init)?;
        println!("{}", text);
        let recovered = serde_json::from_str(&text)?;
        assert_eq!(init, recovered);
        Ok(())
    }

    #[test]
    fn yolo_v5_small_test() {
        let device = Device::cuda_if_available();
        let vs = nn::VarStore::new(device);
        let root = vs.root();
        let input_channels = 3;
        let num_classes = 80;

        let mut yolo_fn = yolo_v5_small(&root, input_channels, num_classes);

        for _ in 0..10 {
            let input = Tensor::randn(
                &[32, input_channels as i64, 224, 224],
                (Kind::Float, Device::cuda_if_available()),
            );
            let instant = std::time::Instant::now();
            let mut output = yolo_fn(&input, false);
            let detections = output.detections();
            let feature_maps = output.feature_maps();

            let feature_map_shapes = feature_maps
                .iter()
                .map(|tensor| tensor.size())
                .collect::<Vec<_>>();
            println!("train output shapes: {:?}", feature_map_shapes);

            {
                let expect = feature_map_shapes
                    .iter()
                    .map(|shape| match shape.as_slice() {
                        &[_bsize, channels, height, width, _outputs] => channels * height * width,
                        _ => unreachable!(),
                    })
                    .sum::<i64>();

                assert_eq!(detections.len(), expect as usize);
            }

            println!("elapsed {:?}", instant.elapsed());
        }
    }
}
