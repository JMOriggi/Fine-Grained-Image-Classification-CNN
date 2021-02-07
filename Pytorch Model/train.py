# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 21:58:09 2020

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
parser.add_argument('--epoch_num', type=int, default=15, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--img_dim', type=int, default=512, help='dimensionality of image feature')
parser.add_argument('--num_categories',type=int,default=1010, help='The number of categories')

parser.add_argument('--DEBUG',type=bool,default=True, help='debug or not')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--save_interval',type=int,default=2,help='the interval between saved epochs')
parser.add_argument('--process_interval',type=int,default=1,help='the interval between process print')

opt = parser.parse_args()
print(opt)



os.makedirs('weights/'+opt.model_name, exist_ok=True)

""" Step1: Configure dataset & model & optimizer """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

traindata = inaturalist2017(phase='val',transform = transform, DEBUG = opt.DEBUG)
valdata = inaturalist2017(phase='val',transform = transform, DEBUG = opt.DEBUG)
train_loader = torch.utils.data.DataLoader(traindata, batch_size=opt.batch_size,shuffle=True,num_workers=opt.n_cpu)
val_loader = torch.utils.data.DataLoader(valdata, batch_size=opt.batch_size,shuffle=True,num_workers=opt.n_cpu)

model = baseline(opt.img_dim, opt.num_categories).to(device)

#optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train(epoch,loadweights=False):
    model.train()
    for batch_idx,[img,label] in enumerate(train_loader):
        img = img.to(device)
        label = label.type(LongTensor)#convert to gpu computation
        
        optimizer.zero_grad()#optimizer.zero_grad() !!!!!!
        output = model(img)

        loss = F.cross_entropy(output, label)
        """nll_loss: negative log likelihood loss"""
        loss.backward()
        optimizer.step()
        
        if batch_idx % opt.process_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * len(label), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        # Tensor.item() to get a Python number from a tensor containing a single value:

def test(epoch):
    with torch.no_grad():
        model.eval()#model.eavl() fix the BN and Dropout
        test_loss = 0
        correct = 0
        try:
            len(test_loader)
        except:
            test_loader = val_loader
        for img,label in test_loader:
            img = img.to(device)
            label = label.type(LongTensor)
            output = model(img)
            
            #sum up batch_loss
            test_loss += F.cross_entropy(output, label)
            #input (Tensor) – (N,C); target (Tensor) – (N)
            
            _, predicted = torch.max(output.data, 1)
            correct += predicted.eq(label).sum().item()
        
#            if batch_idx % 1 == 0:
#                print('Test Epoch: {} [{}/{} ({:.0f}%)]\t'.format(
#                    epoch, batch_idx * len(data), len(test_loader.dataset),
#                    100. * batch_idx / len(test_loader)))
        
        test_loss /= len(test_loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))


for epoch in range(opt.epoch_num):
    train(epoch)
    test(epoch)

    if epoch%opt.save_interval == opt.save_interval-1:
        print('saving the %d epoch' %(epoch+1))
        torch.save(model.state_dict(), "weights/"+opt.model_name+"/epoch-%d.pkl" %(epoch+1))
