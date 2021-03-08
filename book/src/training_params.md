# Training Parameters

## Optimizer

The Adam optimizer is adopted with the following parameters.

- **momentum**: set in `train.json5`
- **weight_decay**: set in `train.json5`
- **beta1**: same as momentum
- **beta2**: 0.999

This is an example optimizer configuration in `train.json5`.

```json5
"training": {
    "optimizer": {
        "momentum": 0.937,
        "weight_decay": 0.0005,
    },
    // ...
}
```

## Learning Rate Scheduling

Currently **constant** and **step-wise** scheduling modes are supported.

The constant scheduling fixes the learning rate to a constant.

```json5
"training": {
    "optimizer": {
        "lr_schedule": {
            "type": "Constant",
            "lr": 0.001,
        },
        // ...
    },
}
```

The step-wise scheduling updates the learning rate after critical training steps. For example, we start by 0.01, set to 1e-3 at step 1200, set to 1e-4 at step 12000, and set to 1e-5 at step 39000.


```json
"training": {
    "optimizer": {
        "lr_schedule": {
            "type": "StepWise",
            "steps": [
                [0, 0.01],
                [1200, 0.001],
                [12000, 0.0001],
                [39000, 0.00001],
            ],
        },
        // ...
    },
}
```

## Loss Function

The loss configuration is the part found in `train.json5`.

```
"training": {
    "loss": {
        "match_grid_method": "Rect4", // Rect2, Rect4
        "box_metric": "Hausdorff",    // IoU, GIoU, DIoU, CIoU, Hausdorff
        "objectness_loss_fn": "Bce",
        "classification_loss_fn": "Bce",
        "objectness_positive_weight": 1.0,
        "iou_loss_weight": 55.0,
        "objectness_loss_weight": 1.0,
        "classification_loss_weight": 55.0,
    },
    // ...
},
```

### `match_grid_method`

The `match_grid_method` is the method to match ground truth boxes to anchor boxes on each feature map. It finds the closest anchor box to each ground truth box, pluses neighboring anchor boxes according to the options.

- `Rect2`: Match up to the top and left neighbor anchor boxes.
- `Rect4`: Match up to the top, left, right and bottom neighbor anchor boxes.

### `box_metric`

The `box_metric` defines the metric to measure the difference between a ground truth box and a matched anchor box. It supports the metrics below.

- IoU ([wikipedia](https://en.wikipedia.org/wiki/Jaccard_index))
- GIoU ([arxiv](https://arxiv.org/abs/1902.09630))
- DIoU ([arxiv](https://arxiv.org/abs/1911.08287))
- CIoU ([arxiv](https://arxiv.org/abs/2005.03572))
- Hausdorff

  It is invented in this project. Please cite _Hsiang-Jui Lin_ and this project for using.


### `objectness_loss_fn`

It defines the metric to measure the objectness performance.

- `Bce`: Binary entropy loss
- `Focal`: Focal loss over binary entropy loss
- `L2`: L^2 distance

### `classification_loss_fn`

It defines the metric to measure the classification performance.

- `Bce`: Binary entropy loss
- `Focal`: Focal loss over binary entropy loss
- `CrossEntropy`: Categorical cross entropy
- `L2`: L^2 distance

### Loss Weights

The total loss is defined as the weighted sum of IoU, objectness and classification losses. The following options sets the weights of each loss term.

- `iou_loss_weight`
- `objectness_loss_weight`
- `classification_loss_weight`
