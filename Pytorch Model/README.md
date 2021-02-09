# Supervised Image Classification Model

## Requirements

The code is implemented by Pytorch (torch 1.5.1, cuda 9.2). Please refer to [Pytorch Tutorials](https://pytorch.org/tutorials/) for more information.

## Data

The iNaturalist2017 dataset can be downloaded from [link](https://github.com/visipedia/inat_comp#data). Please unzip the "train_val2019" folder and put the following files in this folder:

 - `id2img.json `: Dictionary file, map image_id to image_name.
 - `img2id.json` : Dictionary file, map image_name to image_id.
 - `trainlist.txt/vallist.txt/testlist.txt`: text file, each row contains image_name, image_id, category_id



## Training

The model is trained with the following command:

    python train.py

The most relevant arguments are the following:

 - `-model_name `: directory for saving weights
 - `-epoch_num` :number of training epochs
 - `-lr`: learning rate
 - `-DEBUG`: if debug, only a small set of training data will be used

## Evaluation:

The model can be evaluated with the following command:

    python test_submission.py

The most relevant arguments are the following:

 - `-model_name `: directory to load weights
 - `-load_epoch` :specify which epoch to load

Output of testing set is saved as "submission.csv".



