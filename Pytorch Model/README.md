# Supervised Image Classification Model

## Requirements

The code is implemented by Pytorch (torch 1.5.1, cuda 9.2). Please refer to [Pytorch Tutorials](https://pytorch.org/tutorials/) for more information.

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



