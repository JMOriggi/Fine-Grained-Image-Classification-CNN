# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 20:42:15 2020

@author: ZZPzz
"""

import pandas as pd
import numpy as np
import os, sys, glob

import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.color import gray2rgb, rgba2rgb
import skimage.io

import torch
import torchvision.transforms as transforms
import json

from utils import *


FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224), transforms.CenterCrop(224),
                transforms.ToTensor(), normalize
            ])

class inaturalist2017():
    def name(self):
        return "inaturalist2017"
    
    def __init__(self,
                 batch_size: int = 256,
                 file_path: str = './train_val2019/',
                 phase: str = 'train',
                 transform = None,
                 val_split = False,
                 val_ratio = 0.1,
                 DEBUG = False,
                 ):
        super(inaturalist2017, self).__init__()
        self.batch_size = batch_size
        self.phase = phase
        self.val_split = val_split
        self.transform = transform
        
        with open(file_path+phase+'list.txt','r') as f:
            self.data = f.readlines()
        if DEBUG==True:
            self.data = self.data[:2000]
        
        if val_split:
            print("split validation")
            train_len = len(self.save_data) - int(len(self.save_data)/val_ratio)
            print(train_len)
            if self.phase=='train':
                self.data = self.data[:train_len]
            elif self.phase=='val':
                self.data = self.data[train_len:]
                
        
    def __getitem__(self, index: int):
        sample = self.data[index].split("\t")
        
        imgpath = sample[0]
        img_id = sample[1]
        label = int(sample[2])
        
        img_np = skimage.io.imread(imgpath)
        img_np = img_as_ubyte(img_np)
        if self.transform:
            img_np = self.transform(img_np)
        # _,img_np = get_image(imgpath,imsize=224)#(0~1)
        
        if self.phase=='test':
            img_id = int(sample[1])
            return (img_np,img_id,label)
            
        return (img_np, label)
        
    def __len__(self):
        return len(self.data)        
      
    
if __name__=="__main__":
    shuffle=False
    dataset = inaturalist2017(phase='val',transform = transform)
    # dataset = inaturalist2017(phase='val',transform = None)
    
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size = 5,
            num_workers = 0,
            pin_memory = True,
            shuffle = shuffle,
            )
    
    index=0
    print(len(dataloader))
    for batch_idx, (img,label) in enumerate(dataloader):
        print(img.shape,label.shape)
        print(img.type(),label.type())
        if index>=0:
            break
        index+=1
        
    



