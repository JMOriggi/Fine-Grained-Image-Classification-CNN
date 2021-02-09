# Fine Grained Image Classification Neural Network

## Overview

### Supervised Image Classification Model
 Task description
Image classification. Imagine we have cateloged all the plants we care to identify, now we just need to create a classifier for them! Use your skills from the supervised learning sections of this course to try to address this problem.

 Evaluation method
We follow a similar metric to the classification tasks of the ILSVRC. For each image, an algorithm will produce 5 labels. We allow 5 labels because some categories are disambiguated with additional data provided by the observer, such as latitude, longitude and date. It might also be the case that multiple categories occur in an image (e.g. a photo of a bee on a flower). For this competition each image has one ground truth label. For a given image, if the ground truth label is found among the 5 predicted labels, then the error for that image is 0, otherwise it is 1. The final score is the error averaged across all images.

### Semi-Supervised Image Classification Model
 Task description
Semi-Supervised/Few-Shot Learning. Unfortunately, we missed some important plants we want to classify! We do have some images we think contain the plant, but we have only have a few labels. Our new goal is to develop an AI model that can learn from just these labeled examples.

 Evaluation method
We follow classification accuracy. For each image you predict one of the held-out categories as its class. Then we measure how often that guess was correct.

## Data
The iNaturalist2017 dataset can be downloaded from [link](https://github.com/visipedia/inat_comp#data). Please unzip the "train_val2019" folder and put the following files in this folder:
 - `id2img.json `: Dictionary file, map image_id to image_name.
 - `img2id.json` : Dictionary file, map image_name to image_id.
 - `trainlist.txt/vallist.txt/testlist.txt`: text file, each row contains image_name, image_id, category_id


