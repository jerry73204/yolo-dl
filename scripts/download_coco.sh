#!/usr/bin/env bash

mkdir -p coco-2017 &&
    cd coco-2017 &&
    cut -f1 ../coco_links.txt | aria2c --conditional-get true --allow-overwrite true -x8 -i - &&
    cut -f2 ../coco_links.txt | while read file; do echo "unzip '$file'"; done | parallel
