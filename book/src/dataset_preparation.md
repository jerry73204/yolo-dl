# Dataset Preparation

## Microsoft COCO 2017

Download the zips from the following links.

```
http://images.cocodataset.org/zips/train2017.zip
http://images.cocodataset.org/zips/val2017.zip
http://images.cocodataset.org/zips/test2017.zip
http://images.cocodataset.org/zips/unlabeled2017.zip
http://images.cocodataset.org/annotations/annotations_trainval2017.zip
http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
http://images.cocodataset.org/annotations/image_info_test2017.zip
http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip
```

Unzip them to the same target directory. The resulting directory hierarchy looks like this.

```
.
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── ...
├── test2017
│   ├── 000000000001.jpg
│   └── ...
├── train2017
│   ├── 000000000001.jpg
│   └── ...
├── unlabeled2017
│   ├── 000000000001.jpg
│   └── ...
└── val2017
    ├── 000000000001.jpg
    └── ...
```

Configure the `dataset` section in in `train.json5`. Fill the dataset directory in `dataset_dir` option.

```json5
"dataset": {
    "class_whitelist": ["person", "bicycle", "car", "motorcycle", "bus", "truck"], // optional
    "kind": {
        "type": "Coco",
        "image_size": 256,
        "dataset_dir": "/path/to/your/coco/dataset/directory",
        "dataset_name": "train2017",
        "classes_file": "cfg/coco.class",
    },
}
```

## Pascal VOC

Download the dataset from the link corresponding to the year of your interest.

```
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
```

Decompress the TAR file. For example, to extract the VOC 2012 dataset.

```sh
tar -xvf VOCtrainval_11-May-2012.tar
```

The dataset hierarchy will look like this.

```
.
└── VOCdevkit
    └── VOC2012
        ├── Annotations
        ├── ImageSets
        ├── JPEGImages
        ├── SegmentationClass
        └── SegmentationObject

```

Edit the  `train.json5` configuration file. The `dataset` will look like this. The `dataset_dir` points to the `VOC2012/` inside the extracted directory.

```json5
"dataset": {
    "kind": {
        "type": "Voc",
        "image_size": 256,
        "dataset_dir": "/path/to/VOCdevkit/VOC2012",
        "classes_file": "cfg/iii.class",
    },
}
```

## Formosa Dataset

Obtain the Formosa dataset from Institute for Information Industry. You can find the contact in this [link](https://www.iii.org.tw/Product/TransferDBDetail.aspx?tdp_sqno=3345&fm_sqno=23).

The `train.json5` configuration will look like this.

```json5
"dataset": {
    "kind": {
        "type": "Iii",
        "image_size": 256,
        "dataset_dir": "/path/to/your/coco/dataset/directory",
        "classes_file": "cfg/iii.class",
    },
}
```

## CSV Dataset

The CSV dataset consists of a image directory and a label file.

The image directory contains named image files.

```
images
├── cat.jpg
├── dog.jpg
└── pig.jpg
```

The label file is a CSV file which each entry corresponds to an object in an image file. If an image file has more than one objects, there will be multiple entries pointing to the same image file.

```csv
# label.csv
image_file,class_name,cy,cx,h,w
dog.jpg,doggo,203,143.5,288,191
dog.jpg,doggo,206,313,290,146
dog.jpg,doggo,215.5,478,279,180
cat.jpg,meowww,202,327.5,374,619
pig.jpg,piggy,202.5,161.5,361,323
pig.jpg,piggy,139.5,464.0,131,152
```

Edit the `train.json5` configuration file. The `dataset` section will look like this.

```json5
"dataset": {
    "kind": {
        "type": "Csv",
        "image_size": 256,
        "input_channels": 3,
        "image_dir": "/path/to/images",
        "label_file": "/path/to/label.csv",
        "classes_file": "/path/to/classes.txt",
    },
}
```
