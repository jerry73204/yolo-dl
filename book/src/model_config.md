# Model Configuration

## Supported Formats

Currently the following formats are supported with respective features.

- **NEWSLABv1**
  - Se/deserialization
  - Shape inspection
  - Export computation graph
  - Model training

- **darknet**
  - Se/deserialization
  - Shape inspection
  - Export computation graph

The model training using **darknet** configuration is not implemented yet. It's still work in progress.

The **NEWSLABv1** is our replacement to **darknet**. It features modular design and more versatile shape inference.


## The NEWSLABv1 Model Description Format

### The Model Format

The model file is written in [JSON5](https://json5.org/), an extension to JSON that allows comments. It consists of a set of _groups_ and the _name_ of main group.

```json5
// model.json5
{
    "main_group": "Main",
    "groups": {
        "Main": [
            { /* layer 1 */ },
            { /* layer 2 */ },
            { /* layer 3 */ },
            ...
        ],
        "SubGroup1": [
            // layers
        ],
        "SubGroup2": [
            // layers
        ],
        ...
    }
}
```

### The Layer Format

Each layer has the following common fields.

- **kind**: The type of the layer.
- **name** (optional): The name of the layer.
- **from** (optional): The name of the layer to take the input. If it is not specified, it assumes the previous layer.

The example has two paths forking from the input layer, then are summed up together, forming a diamond shaped model.

```json5
    "groups": {
        "Main": [
            // input
            {
                "name": "input",
                "kind": "Input",
                "shape": ["_", 3, 512, 512],
            },

            // branch 1
            {
                // Here "from" is not specified, assuming the previous input layer.
                "kind": "ConvBn2D",
                "name": "branch1",
                "c": 256,
                "k": 3,
                "s": 2,
            },

            // branch 2
            {
                // The "from" is explicitedly specified.
                "kind": "ConvBn2D",
                "name": "branch2",
                "from": "input",
                "c": 256,
                "k": 3,
                "s": 2,
            },

            // sum
            {
                // Sum2D is a multi-input layer. The "from" accepts a list of names here.
                "kind": "Sum2D",
                "from": ["branch1", "branch2"],
            },
        ]
    }
```

### Groups

To import the module defined in another group, the `GroupRef` layer describes the group to be imported and the input layers fed to the group.

In the example below, it declares a `MyModule` group with an input and an output layer respectively named `sub_input`, `sub_output`. The `Main` group connects the `my_input` layer to `MyModule`'s `sub_input`, and then takes the output from `sub_output` layer in group.

```json5
// model.json5
{
    "main_group": "Main",
    "groups": {
        "Main": [
            {
                "kind": "ConvBn2D",
                "name": "my_input",
                "c": 256,
                "k": 3,
                "s": 2,
            },
            {
                "kind": "GroupRef",
                "name": "subroutine",
                // wires "my_input" to group's "sub_input"
                "from": {
                    "sub_input": "my_input",
                },
                "group": "MyModule",
            },
            {
                "kind": "ConvBn2D",
                // receive from "sub_output" layer from the group named "sub" here.
                "from": "sub.sub_output",
                "c": 256,
                "k": 3,
                "s": 2,
            },
            ...
        ],
        "MyModule": [
            {
                "name": "sub_input",
                ...
            }
            ...
            {
                "name": "sub_output",
                ...
            }
        ],
        ...
    }
}
```

### Includes

A set of groups can be packed into an individual _group unit_ file.

```json5
// groups.json5
{
    "groups": {
        "Generator": [ /* layers */ ],
        "Discriminator": [ /* layers */ ],
        "Loss": [ /* layers */ ],
    }
}
```

We add the groups unit file to `includes` list in the model configuration, so that we can reuse the groups from that file.

```json5
// model.json5
{
    "main_group": "Main",
    "includes": ["groups.json5"],
    "groups": {
        "Main": [
            { /* layer 1 */ },
            { /* layer 2 */ },
            { /* layer 3 */ },
            ...
        ],
        ...
    }
}
```


## Inspect a NEWSLABv1 Configuration File

NEWSLABv1 is a model description format used by NEWSLAB. To inspect a configuration file `model-config/tests/cfg/yolov4-csp.json5` for example,

```sh
cargo run --release --bin model-config -- \
    info model-config/tests/cfg/yolov4-csp.json5
```

It can plot the model architecture into an image. For export an SVG file,


```sh
// export GraphViz DOT file
cargo run --release --bin model-config -- \
    make-dot-file \
    model-config/tests/cfg/yolov4-csp.json5 \
    image.dot

// convert DOT to SVG
dot -Tsvg image.dot > image.svg
```

## Inspect a Darknet Configuration File

The `darknet-config` is a toolkit to inspect configuration files from darknet project ([link](https://github.com/AlexeyAB/darknet)). To show the model information,

```sh
cargo run --release --bin darknet-config -- \
    info yolov4.cfg
```

The `make-dot-file` subcommand can plot the computation graph.

```sh
// export GraphViz DOT file
cargo run --release --bin darknet-config -- \
    make-dot-file yolov4.cfg image.dot

// convert DOT to SVG
dot -Tsvg image.dot > image.svg
```
