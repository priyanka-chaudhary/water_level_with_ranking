from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import utils, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from skimage import io, transform
from PIL import Image
import json
import math


from torch.utils.data.dataset import Dataset

class OneDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, img_path, labels, transforms = None):
        'Initialization'
        #self.image_ids = image_ids
        self.labels = labels
        self.img_path = img_path
        self.transforms = transforms


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_path)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        #image_id = self.image_ids[index]
        #print("Image id: ",image_id)

        # Load data and get label
        X = Image.open(self.img_path[index])
        #print("Image channels: ", X.layers)
        if hasattr(X, 'layers'):
            if X.layers != 3:
                #print("Not 3 channels: ", X.layers)
                pass
        else:
            print("Doesnt have layers")
        if X.mode == 'CMYK':
            X = X.convert('RGB')
        y = float(self.labels[index])
        #print(y)

        if self.transforms is not None:
            X = self.transforms(X)

        return X, y

#level to cm table
level_ref = np.array([[0.0,0.0],
         [0.0,1.0],
         [1.0,10.0],
         [10.0, 21.25],
         [21.25,42.5],
         [42.5,63.75],
         [63.75,85.0],
         [85.0,106.25],
         [106.25,127.25],
         [127.25,148.75],
         [148.75,170.0]])


# level to cm conversion
def level_to_cm(level_ref, level):
    floor = int(math.floor(level))
    #print(floor)
    ceil = int(math.ceil(level))
    #print(ceil)
    percent = level - floor

    level_cm = (level_ref[ceil][1] - level_ref[ceil][0]) * percent + level_ref[floor][1]
    return level_cm

params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 4}

data_transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225])
    ])

##############################################################################

val_img_ids =[]
val_img_labels = []

with open('/cluster/work/igp_psr/pchaudha/flood/rank/k/k2/gt_val_reg.json') as f:
    val = json.load(f)

for key, value in val.items():
    # add image path to load it later
    val_img_ids.append(str(key))

    # add image label
    val_img_labels.append(value)
    #print(key, value)

with open('/cluster/work/igp_psr/pchaudha/flood/rank/k/k2/gt_val_rank.json') as f:
    val = json.load(f)

for key, value in val.items():
    # add image path to load it later
    val_img_ids.append(str(key))

    # add image label
    val_img_labels.append(value)
    #print(key, value)


#instantiate dataset class
val2 = OneDataset(img_path=val_img_ids, labels=val_img_labels, transforms=data_transform)
val_generator = torch.utils.data.DataLoader(val2,**params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model=torch.load('/scratch/pchaudha/ranking_loss/runs/rank/rank_trial_1.pth')
model=torch.load('/cluster/work/igp_psr/pchaudha/flood/rank/exp/change_batch_size/change_batch_size_6/runs/fold2/rank_model_fold2.pth')
model.eval()

error = 0.0
error_level = 0.0
test_size = len(val_img_labels)
for i, x in enumerate(val_generator):
    #x = x.float()
    input = x[0]
    gt = x[1]
    input = input.to(device)
    y_pred = model(input)
    y = y_pred.data.cpu().numpy()
    pred_cms = level_to_cm(level_ref,float(abs(y)))
    #y_gt = test_labels[i]
    y_gt = gt.cpu().numpy()
    gt_cms = level_to_cm(level_ref, y_gt[0])
    temp = (gt_cms - pred_cms)
    temp = temp**2
    t = math.sqrt(temp)

    p = float(y_gt[0])-float(y[0])
    p1 = p**2
    p2 = math.sqrt(p1)

    error_level = error_level + p2

    error = error + t #abs(gt_cms - pred_cms)
error_test = error/test_size
level_test = error_level/test_size
print(" RMSE error on val set in cms is: ", error_test)
print(" RMSE error on val set in level is: ", level_test)

##############################################################################
#load test data from json file
test_img_path = []
test_labels = []
test_image_ids = []

# check file is json
# with open('imageList_test.json') as f:
with open('/cluster/work/igp_psr/pchaudha/flood/rank/k/k2/gt_test.json') as f:
    train = json.load(f)

for key, value in train.items():
    # add image path to load it later
    test_img_path.append(str(key))

    # add image label
    test_labels.append(value)
    #print(key, value)

#instantiate dataset class
test = OneDataset(img_path=test_img_path, labels=test_labels, transforms=data_transform)
test_generator = torch.utils.data.DataLoader(test,**params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model=torch.load('/scratch/pchaudha/ranking_loss/runs/rank/rank_trial_1.pth')
model=torch.load('/cluster/work/igp_psr/pchaudha/flood/rank/exp/change_batch_size/change_batch_size_6/runs/fold2/rank_model_fold2.pth')
model.eval()

error = 0.0
error_level = 0.0
test_size = len(test_labels)
for i, x in enumerate(test_generator):
    #x = x.float()
    print("___________________________________")
    print("Image id: ",test_img_path[i])
    input = x[0]
    gt = x[1]
    input = input.to(device)
    y_pred = model(input)
    y = y_pred.data.cpu().numpy()
    print("Prediction: ", y[0])
    pred_cms = level_to_cm(level_ref,float(abs(y)))
    #y_gt = test_labels[i]
    y_gt = gt.cpu().numpy()
    print("Gt: ", y_gt)
    gt_cms = level_to_cm(level_ref, y_gt[0])
    temp = (gt_cms - pred_cms)
    temp = temp**2
    t = math.sqrt(temp)

    p = float(y_gt[0])-float(y[0])
    p1 = p**2
    p2 = math.sqrt(p1)

    error_level = error_level + p2

    error = error + t #abs(gt_cms - pred_cms)
error_test = error/test_size
level_test = error_level/test_size
print(" RMSE error on test in cms is: ", error_test)
print(" RMSE error on test in level is: ", level_test)
print("Done")