# The Project Architecture

## Crates

Algoirthms and utility crates.

- **tch-goodies**: Extensions to tch-rs.
- **tch-nms**: The NMS algorithm written in CUDA.

Model configuration crates.

- **darknet-config**: The Darknet configuration file parser and data types.
- **model-config**: The NEWSLABv1 model configuration file parser and data types.

Image detection model and training.

- **yolo-dl**: Model building blocks and data processing units for image detection.
- **train**: The model training program.
