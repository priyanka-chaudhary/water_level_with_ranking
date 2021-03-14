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
from itertools import combinations
from model import PreeNet
from itertools import cycle
import itertools
import datetime

from tensorboardX import SummaryWriter
import flag

n_class = 11

plt.ion()   # interactive mode

def myCycle(iterable):
    while True:
        for x in iterable:
            yield x

#from torch.utils import data
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

class PairDataset(Dataset):
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

      # Load data and get label
      X = Image.open(self.img_path[index])
      # print("Image channels: ", X.layers)
      if hasattr(X, 'layers'):
          if X.layers != 3:
              # print("Not 3 channels: ", X.layers)
              pass
      else:
          print("Doesnt have layers")
      if X.mode == 'CMYK':
          X = X.convert('RGB')
      y = float(self.labels[index])
      # print(y)

      if self.transforms is not None:
          X = self.transforms(X)

      return X, y

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


if __name__ == '__main__':

    # initialize the network
    model = models.vgg16(pretrained='imagenet')

    # remove the top layers
    #model.features = nn.Sequential(*(model.features[i] for i in range(30)))
    #model.classifier = nn.Sequential(*list(model.children())[:-3])
    #model.avgpool = nn.AdaptiveAvgPool2d(32)

    #model_reg, model_rank = PreeNet(model)

    #model_reg.cuda()
    #model_rank.cuda()

    # Since imagenet as 1000 classes , We need to change our last layer according to the number of classes we have,
    model.classifier._modules['3'] = nn.Linear(4096, 1000)
    model.classifier._modules['6'] = nn.Linear(1000, 1)

    model_upd = PreeNet(model)#, rank_x1, rank_x2)

    model.cuda()

    #tensorboard summary
    x = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    y = str.split(x)
    folder_name = y[0]+"_"+y[1]
    path_loc = "/cluster/work/igp_psr/pchaudha/flood/rank/exp/change_batch_size/change_batch_size_2/runs/fold2/"
    #writer = SummaryWriter('runs/rank_26_06_19/')
    writer = SummaryWriter(path_loc)

    #removing last layer from resnet
    #model = nn.Sequential(*list(model.children())[:-1])

    train_img_path_1 = []
    train_labels_1 = []
    train_image_ids_1 = []
    train_img_path_2 = []
    train_labels_2 = []
    train_image_ids_2 = []

    val_img_path_1 = []
    val_labels_1 = []
    val_image_ids_1 = []
    val_img_path_2 = []
    val_labels_2 = []
    val_image_ids_2 = []

    # check file is json
    # Regression task
    with open('/cluster/work/igp_psr/pchaudha/flood/rank/k/k2/gt_train_reg.json') as f:
        train = json.load(f)
    strJson = json.dumps(train)

    for key, value in train.items():
        # add image path to load it later
        train_img_path_1.append(str(key))

        # add image label
        train_labels_1.append(value)
        print(key, value)

    with open('/cluster/work/igp_psr/pchaudha/flood/rank/k/k2/gt_train_rank.json') as f:
        train = json.load(f)
    strJson = json.dumps(train)

    for key, value in train.items():
        # add image path to load it later
        train_img_path_2.append(str(key))

        # add image label
        train_labels_2.append(value)
        print(key, value)

    # check file is json
    with open('/cluster/work/igp_psr/pchaudha/flood/rank/k/k2/gt_val_reg.json') as v1:
        val = json.load(v1)
    strJson = json.dumps(val)

    for k, v in val.items():
        # add image path to load it later
        val_img_path_1.append(str(k))

        # add image label
        val_labels_1.append(v)
        print(k, v)

    # check file is json
    with open('/cluster/work/igp_psr/pchaudha/flood/rank/k/k2/gt_val_rank.json') as v2:
        val2 = json.load(v2)

    for path, label in val2.items():
        # add image path to load it later
        val_img_path_2.append(str(path))

        # add image label
        val_labels_2.append(label)
        print(path, label)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    #have to check for this flag
    #torch.bakends.cudnn.benchmark = True

    # Parameters
    params_rank = {'batch_size': 18,
              'shuffle': True,
              'drop_last': True,
              'num_workers': 4}

    params_reg = {'batch_size': 2,
              'shuffle': True,
              'drop_last': True,
              'num_workers': 4}

    max_epochs = 400

    data_transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225])
    ])

    #instantiate dataset class
    train1 = OneDataset(img_path=train_img_path_1, labels=train_labels_1 , transforms=data_transform)
    train2 = PairDataset(img_path=train_img_path_2, labels=train_labels_2 , transforms=data_transform)

    validation1 = OneDataset(img_path=val_img_path_1, labels=val_labels_1 , transforms=data_transform)
    validation2 = PairDataset(img_path=val_img_path_2, labels=val_labels_2 , transforms=data_transform)

    train_one = torch.utils.data.DataLoader(train1, **params_reg)
    train_pair = torch.utils.data.DataLoader(train2, **params_rank)
    val_one = torch.utils.data.DataLoader(validation1, **params_reg)
    val_pair = torch.utils.data.DataLoader(validation2, **params_rank)


    dataloaders = {
        'train': [train_one, train_pair],
        'validation': [val_one, val_pair]
    }

    #TODO
    #Calculate the length of datasets here
    data_length = {
        'train': [len(train1),len(train2)],
        'validation' : [len(validation1),len(validation2)]
    }

    #criterion, optimizer
    #loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    loss_func = torch.nn.MSELoss()
    criterion = torch.nn.MarginRankingLoss(margin=0)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180], gamma=0.1)


    #Training
    def train_model(model, loss_func, optimizer, scheduler, num_epochs=200):

        since = time.time()

        val_loss_history = []
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = np.Inf

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            for phase in ['train', 'validation']:
                if phase == 'train':
                    scheduler.step()
                    model.train()
                    #print("Train")
                else:
                    model.eval()
                    #print("Validation")

                running_loss1 = 0.0
                running_loss2 = 0.0
                running_corrects = 0

                #Note that if x and y are not the same length, zip will truncate to the shortest list.
                #if you don't want to truncate to the shortest list, you could use itertools.zip_longest
                #CHECK_LATER
                count_batches = 0
                #for i, data in enumerate(zip(dataloaders[phase][0], dataloaders[phase][1])):
                for i, data in enumerate(zip(itertools.cycle(dataloaders[phase][0]), dataloaders[phase][1])):
                #for i, data in enumerate(itertools.zip_longest(myCycle(dataloaders[phase][0]), dataloaders[phase][1])):

                    #print(data)
                    count_batches = count_batches + 1
                    #print(count_batches)
                    dataset_1 = data[0]
                    dataset_2 = data[1]

                    x_1 = dataset_1[0]
                    y_1 = dataset_1[1]
                    x_2 = dataset_2[0]
                    y_2 = dataset_2[1]
                    #y_2 = y_2.numpy()
                    y_2_list = y_2.squeeze().tolist()

                    #make pairs from batch of pair dataset
                    y = 0
                    gt_list = []
                    #gt_list.append((b*(b-1))/2)
                    for i in range(params_rank['batch_size']):
                        for j in range(i+1,params_rank['batch_size']):
                            #input1 = x_2[i]
                            #input2 = x_2[j]

                            gt1 = y_2_list[i]
                            gt2 = y_2_list[j]

                            if gt1 > gt2:
                                y = 1
                            elif gt2 > gt1:
                                y = -1
                            else:
                                y = 0
                            gt_list.append(y)

                    labels_numpy = y_1.numpy()
                    labels_numpy = [[x] for x in labels_numpy]
                    labels_numpy = np.array(labels_numpy)
                    target1 = torch.from_numpy(labels_numpy)

                    input1 = x_1.to(device)
                    target1 = target1.to(device, dtype=torch.float)

                    input2 = x_2.to(device)
                    target2 = torch.FloatTensor(gt_list)
                    target2 = target2.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        flag.TRAIN_RANK = False
                        #print("train_flag: ", flag.TRAIN_RANK)
                        outputs = model(input1)
                        _, preds = torch.max(outputs, 1)
                        loss1 = loss_func(outputs, target1)

                        flag.TRAIN_RANK = True
                        #print("train_flag: ", flag.TRAIN_RANK)
                        x1, x2 = model(input2)
                        loss2 = criterion(x1, x2, target2)


                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss = loss1 + (flag.LAMBDA * ( loss2 ))
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss1 += loss1.item() * input1.size(0)
                    #N_running_loss1 += input1.size(0)
                    running_loss2 += loss2.item() * x2.size(0)# number of pair
                    #N_running_loss2 += x2.size(0)

                #running_corrects += torch.sum(preds == labels.data)

                #epoch_loss = running_loss / len(image_datasets[phase])
                #epoch_loss = (running_loss1 / data_length[phase][0]) + (running_loss2 / count_batches)
                b = params_rank['batch_size']
                #to calculate number of pairs generated from whole dataset
                n_samples = count_batches*(b)*(b-1)/2
                epoch_loss = (running_loss1 / data_length[phase][0]) + flag.LAMBDA*((running_loss2 / n_samples) )
                reg_loss = (running_loss1 / data_length[phase][0])
                rank_loss = flag.LAMBDA*((running_loss2 / n_samples))
                #epoch_acc = running_corrects.double() / len(image_datasets[phase])

                print('{} loss: {:.4f}'.format(phase, epoch_loss))
                if phase == 'train':
                    writer.add_scalar('reg_loss_train', reg_loss, epoch)
                    writer.add_scalar('rank_loss_train', rank_loss, epoch)
                    writer.add_scalar('train_loss', epoch_loss, epoch)
                elif phase == 'validation':
                    writer.add_scalar('reg_loss_val', reg_loss, epoch)
                    writer.add_scalar('rank_loss_val', rank_loss, epoch)
                    writer.add_scalar('val_loss', epoch_loss, epoch)
                    #for scheduler reduceLROnPlateau
                    #scheduler.step(epoch_loss)

                # deep copy the model
                if phase == 'validation' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    model_p = path_loc + "/rank_model_fold2.pth"
                    torch.save(model,model_p)
                    print("Epoch is: ", epoch)
                    print("Epoch loss is: ", epoch_loss)
                if phase == 'validation':
                    val_loss_history.append(epoch_loss)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        # load best model weights
        model.load_state_dict(best_model_wts)
        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()
        return model, val_loss_history


    model_trained, hist = train_model(model_upd, loss_func, optimizer, scheduler, num_epochs=200)
    print("Done")

