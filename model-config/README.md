# The NEWSLABv1 Model Configuration Format

It project implements NEWSLABv1 deep learning architecture description format. It has the following features:

- **Modular**: A set of layers can be packed into reusable groups.
- **Safety**: The parser ensures each layer receives valid tensor shapes.
- **Versatility**: The format allows multi-input/output and heterogeneous shape models.

## Format Description

### The Model Format

The model file is written in [JSON5](https://json5.org/). It consists of a set of _groups_ and the name of main group.

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

## Command Line Usage

### Plot Computation Graph

The command plots the computation graph of a model file in a [GraphViz](https://graphviz.org/) DOT file.

```sh
cargo run --bin model-config \
    make-dot-file \
    model-config/tests/cfg/yolov4-csp.json5 \
    output.dot
```

Then, transform the DOT file to SVG file. You can change the `-Tsvg` option to `-Tpng` to export a PNG image.

```sh
dot -Tsvg output.dot > output.svg
```
