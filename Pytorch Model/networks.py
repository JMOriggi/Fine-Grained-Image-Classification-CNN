# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 21:43:01 2020

@author: ZZPzz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def DensewithBN(in_fea, out_fea, normalize=True):
    layers=[nn.Linear(in_fea, out_fea)]
    if normalize==True:
        layers.append(nn.BatchNorm1d(num_features = out_fea))
    layers.append(nn.ReLU())
    return layers

class baseline(nn.Module):
    def __init__(self,
                 image_dim: int = 512,
                 num_category: int = 1010,
                 ):
        super(baseline, self).__init__()
        self.image_dim = image_dim
        
        # res18 = torchvision.models.resnet18(pretrained=True)
        res18 = torchvision.models.resnet18(pretrained=False)
        # print(res18)
        self.img_encoder = nn.Sequential(*list(res18.children())[:-1])
        self.imgdense = nn.Linear(image_dim, num_category)
        
    def forward(self, img):        
        batch_size = img.shape[0]
        img = self.img_encoder(img) # (None,512,1,1)
        output = self.imgdense(img.view(batch_size,-1))
        return output

if __name__=="__main__":
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 12
    _input = torch.randn(batch_size,3,224,224)
    _input = _input.type(FloatTensor)
    
    # txt_tensor = txt_tensor.type(LongTensor)
    # binary_txt = txt_tensor

    model = baseline().to(device)
    output = model(_input)
    
    print(output.shape)
    
    