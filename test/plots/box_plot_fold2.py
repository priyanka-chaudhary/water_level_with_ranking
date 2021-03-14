from __future__ import print_function, division

import matplotlib

matplotlib.use('Agg')

import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
         [148.75,170.0],
         [170.0,170.0]])


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
    print(key, value)

#instantiate dataset class
test = OneDataset(img_path=test_img_path, labels=test_labels, transforms=data_transform)
test_generator = torch.utils.data.DataLoader(test,**params)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model=torch.load('/scratch/pchaudha/ranking_loss/runs/rank/rank_trial_1.pth')
model=torch.load('/cluster/work/igp_psr/pchaudha/flood/rank/exp/three_pair_loader/three_pair_loader_2/runs/fold5/rank_model_fold2.pth')
model.eval()

ground_truth = [[] for i in range(11)]
error = 0
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
    if y_gt[0]==0:
        ground_truth[0].append(float(y[0]))
    elif y_gt[0]==1:
        ground_truth[1].append(float(y[0]))
    elif y_gt[0]==2:
        ground_truth[2].append(float(y[0]))
    elif y_gt[0]==3:
        ground_truth[3].append(float(y[0]))
    elif y_gt[0]==4:
        ground_truth[4].append(float(y[0]))
    elif y_gt[0]==5:
        ground_truth[5].append(float(y[0]))
    elif y_gt[0]==6:
        ground_truth[6].append(float(y[0]))
    elif y_gt[0]==7:
        ground_truth[7].append(float(y[0]))
    elif y_gt[0]==8:
        ground_truth[8].append(float(y[0]))
    elif y_gt[0]==9:
        ground_truth[9].append(float(y[0]))
    elif y_gt[0]==10:
        ground_truth[10].append(float(y[0]))
    else:
        print("Not possible")
    temp = abs(gt_cms - pred_cms)
    temp = temp**2
    t = math.sqrt(temp)
    error = error + t #abs(gt_cms - pred_cms)
error_test = error/test_size
print("Error on test set is: ", error_test)


# Usual boxplot
import seaborn as sns
import operator as op

sorted_keys = ['lev0', 'lev1', 'lev2', 'lev3', 'lev4', 'lev5', 'lev6', 'lev7', 'lev8', 'lev9', 'lev10']

fig = plt.figure(figsize=(20, 10))
sns.set_style("darkgrid")
sns.utils.axlabel(xlabel="Ground truth labels", ylabel="Prediction values", fontsize=20)
sns.boxplot(data=ground_truth)
sns.swarmplot(data=ground_truth, edgecolor="black")
plt.title("Prediction plot for test images", fontsize=22)
plt.xticks(plt.xticks()[0], sorted_keys, fontsize=18)
plt.yticks(np.arange(0,11,step=1))
plt.savefig('/scratch/pchaudha/ranking_loss/plots/box_plot/boxplot_with_points_f2.png', bbox_inches='tight', dpi=1200)
plt.savefig('/scratch/pchaudha/ranking_loss/plots/box_plot/boxplot_with_points_f2.pdf', bbox_inches='tight', dpi=1200)
plt.clf()
#plt.show()

print("Done")