# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 23:42:57 2020

@author: ZZPzz
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms


import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from dataset import *
from networks import baseline


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='baseline', help='name of the trained model')
parser.add_argument('--load_epoch', type=int, default=10, help='epoch to load')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
# parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--img_dim', type=int, default=512, help='dimensionality of image feature')
parser.add_argument('--num_categories',type=int,default=1010, help='The number of categories')

parser.add_argument('--DEBUG',type=bool,default=True, help='debug or not')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--process_interval',type=int,default=2,help='the interval between process print')


opt = parser.parse_args()
print(opt)
os.makedirs('weights/'+opt.model_name, exist_ok=True)


""" Step1: Configure dataset & model & optimizer """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

testdata = inaturalist2017(phase='test',transform = transform, DEBUG = opt.DEBUG)
test_loader = torch.utils.data.DataLoader(testdata, batch_size=opt.batch_size,shuffle=False,num_workers=opt.n_cpu)

model = baseline(opt.img_dim, opt.num_categories).to(device)
model.load_state_dict(torch.load("weights/"+opt.model_name+"/epoch-%d.pkl" %(opt.load_epoch)))

def test(epoch, test_loader=test_loader):
    with torch.no_grad():
        model.eval()#model.eavl() fix the BN and Dropout
        test_loss = 0
        correct = 0
        try:
            len(test_loader)
            print("aaa")
        except:
            test_loader = val_loader
        
        prediction = []
        img_ids = []
        for batch_idx,[img,img_id,label] in enumerate(test_loader):
            img = img.to(device)
            label = label.type(LongTensor)
            output = model(img)
            
            #sum up batch_loss
            test_loss += F.cross_entropy(output, label)
            #input (Tensor) – (N,C); target (Tensor) – (N)
            
            _, predicted = torch.max(output.data, 1)
            prediction.append(predicted.cpu().numpy())
            img_ids.append(img_id.numpy())
            correct += predicted.eq(label).sum().item()
            
            if batch_idx % opt.process_interval == 0:
                print('Test Epoch {}: [{}/{} ({:.0f}%)]\t'.format(
                    epoch, batch_idx * opt.batch_size, len(test_loader.dataset),
                    100. * batch_idx / len(test_loader)))
        
        test_loss /= len(test_loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))
        return prediction, img_ids

prediction,img_ids = test(opt.load_epoch)

test_id = np.hstack(img_ids)
test_label = np.hstack(prediction)

save_csv(test_id,test_label,filename='submission.csv')


