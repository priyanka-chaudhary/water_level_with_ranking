#Takes mask r-cnn imagelist files and gt_train_new and gt_val_new and cross references which images
#we used in mask-rcnn task and rest.
#The mask r-cnn used images would be used for regression task and remaining for rank task.

import numpy as np
import json
import itertools

#Find the image numbers in the imageList_train and val
masked_file = []
with open("/scratch/pchaudha/mask-rcnn/Mask_RCNN-master/dataset-coco/imageList_train.txt", "r") as ins:
    for line in ins:
        line = line.strip()
        if line.endswith('.png'):
            line = line[:-4]
        masked_file.append(line)

with open("/scratch/pchaudha/mask-rcnn/Mask_RCNN-master/dataset-coco/imageList_val.txt", "r") as ins:
    for line in ins:
        line = line.strip()
        if line.endswith('.png'):
            line = line[:-4]
        masked_file.append(line)

total_ids = []
total_path = "/scratch/pchaudha/Images/After cleaning/dataset/FloodImages_newname/"
total_labels = []

# check file is json
# with open('imageList_test.json') as f:
with open('gt_train_new.json') as f:
    train = json.load(f)
strJson = json.dumps(train)

for key, value in train.items():
    # add image path to load it later
    #train_img_path_1.append(str(key))

    # add image id to image ids
    if "Flood_" in key:
        #temp = key.replace('Flood_', ' ')
        temp = key
        #temp.startswith()
        pass
    if "Original/" in key:
        continue
    temp = temp.replace('.jpg', ' ')
    image_id = temp.split("/")
    image_id = image_id[7]
    image_id = image_id.strip()
    total_ids.append(image_id)

    # add image label
    total_labels.append(value)
    print(key, value)

# check file is json
# with open('imageList_test.json') as f:
with open('gt_val_new.json') as f:
    train = json.load(f)
strJson = json.dumps(train)

for key, value in train.items():
    # add image path to load it later
    #train_img_path_1.append(str(key))

    # add image id to image ids
    if "Flood_" in key:
        #temp = key.replace('Flood_', ' ')
        temp = key
        pass
    if "Original/" in key:
        continue
    temp = temp.replace('.jpg', ' ')
    image_id = temp.split("/")
    image_id = image_id[7]
    image_id = image_id.strip()
    total_ids.append(image_id)

    # add image label
    total_labels.append(value)
    print(key, value)

rank_id = []
rank_label = []
reg_id = []
reg_label = []
for id in masked_file:
    if id in total_ids:
        idx = total_ids.index(id)
        path = total_path + str(id) + ".jpg"
        reg_id.append(path)
        reg_label.append(total_labels[idx])

    else:
        path = total_path + str(id) + ".jpg"
        reg_id.append(path)
        t = "annotate me"
        reg_label.append(t)

for img in total_ids:
    if img in masked_file:
        pass
    else:
        path = total_path + str(img) + ".jpg"
        rank_id.append(path)
        idx = total_ids.index(img)
        rank_label.append(total_labels[idx])

reg_dict = dict(zip(reg_id, reg_label))
rank_dict = dict(zip(rank_id, rank_label))

with open('gt_reg.json', 'w') as fp:
    json.dump(reg_dict, fp, sort_keys = True, indent=4)

with open('gt_rank.json', 'w') as fq:
    json.dump(rank_dict, fq, sort_keys = True, indent=4)

print("Done")