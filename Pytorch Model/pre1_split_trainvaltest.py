# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 20:46:06 2020

@author: ZZPzz
"""
import json
import numpy as np

with open("rawData/train2019.json",'r') as f:
    train = json.load(f)
    train_images = train['images']
    train_anno = train['annotations']
    
with open("rawData/val2019.json",'r') as f:
    val = json.load(f)
    val_images = val['images']
    val_anno = val['annotations']


img2id = {}
id2img = {}
for image in train_images+val_images:
    img_name = image['file_name']#.split("/")[-1]
    
    if img_name not in img2id:
        img2id[img_name] = image['id']
        id2img[image['id']] = img_name
    else:
        print("error")

with open("train_val2019/img2id.json",'w') as f:
    json.dump(img2id,f)
with open("train_val2019/id2img.json",'w') as f:
    json.dump(id2img,f)

with open("train_val2019/img2id.json",'r') as f:
    img2id = json.load(f)
with open("train_val2019/id2img.json",'r') as f:
    id2img = json.load(f)

annos = train_anno+val_anno

np.random.seed(0)
np.random.shuffle(annos)

total_len = len(annos)
# print(annos[0])
print("train_val2019set length: ",total_len)


test_len = 30000 # inaturalist testing set: 35350
val_len = int(total_len/10)
train_len = total_len-val_len-test_len

with open("train_val2019/trainlist.txt",'w') as f:
    for anno in annos[:train_len]:
    # for anno in annos[:10]:
        f.write(id2img[str(anno['id'])]+"\t"+str(anno['image_id'])+"\t"+str(anno['category_id'])+"\n")
        
with open("train_val2019/vallist.txt",'w') as f:
    for anno in annos[train_len:train_len+val_len]:
        f.write(id2img[str(anno['id'])]+"\t"+str(anno['image_id'])+"\t"+str(anno['category_id'])+"\n")

with open("train_val2019/testlist.txt",'w') as f:
    for anno in annos[train_len+val_len:]:
        f.write(id2img[str(anno['id'])]+"\t"+str(anno['image_id'])+"\t"+str(anno['category_id'])+"\n")





