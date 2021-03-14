
import os
import glob
import json
from collections import Counter
import random
import math

reg_img_id = []
reg_img_label = []

with open('gt_rank_complete_new.json') as f:
    reg = json.load(f)

for key, value in reg.items():
    # add image path to load it later
    reg_img_id.append(str(key))

    # add image label
    reg_img_label.append(value)
    print(key, value)

print(Counter(reg_img_label))

#make subset for various level values
id_0 = [k for k,v in reg.items() if int(v) == 0]
id_1 = [k for k,v in reg.items() if int(v) == 1]
id_2 = [k for k,v in reg.items() if int(v) == 2]
id_3 = [k for k,v in reg.items() if int(v) == 3]
id_4 = [k for k,v in reg.items() if int(v) == 4]
id_5 = [k for k,v in reg.items() if int(v) == 5]
id_6 = [k for k,v in reg.items() if int(v) == 6]
id_7 = [k for k,v in reg.items() if int(v) == 7]
id_8 = [k for k,v in reg.items() if int(v) == 8]
id_9 = [k for k,v in reg.items() if int(v) == 9]
id_10 = [k for k,v in reg.items() if int(v) == 10]

#shuffle all the lists
random.Random(19).shuffle(id_0)
random.Random(19).shuffle(id_1)
random.Random(19).shuffle(id_2)
random.Random(19).shuffle(id_3)
random.Random(19).shuffle(id_4)
random.Random(19).shuffle(id_5)
random.Random(19).shuffle(id_6)
random.Random(19).shuffle(id_7)
random.Random(19).shuffle(id_8)
random.Random(19).shuffle(id_9)
random.Random(19).shuffle(id_10)

val_0 = random.sample(id_0,math.ceil(0.22*len(id_1)))
val_1 = random.sample(id_1,math.ceil(0.22*len(id_1)))
val_2 = random.sample(id_2,math.ceil(0.22*len(id_2)))
val_3 = random.sample(id_3,math.ceil(0.22*len(id_3)))
val_4 = random.sample(id_4,math.ceil(0.22*len(id_4)))
val_5 = random.sample(id_5,math.ceil(0.22*len(id_5)))
val_6 = random.sample(id_6,math.ceil(0.22*len(id_6)))
val_7 = random.sample(id_7,math.ceil(0.22*len(id_7)))
val_8 = random.sample(id_8,math.ceil(0.22*len(id_8)))
val_9 = random.sample(id_9,math.ceil(0.22*len(id_9)))
val_10 = random.sample(id_10,math.ceil(0.22*len(id_10)))

#combine all val_id lists to get the validation set
val_ids = val_0 + val_1 +val_2 + val_3 + val_4 + val_5 + val_6 + val_7 + val_8 + val_9 + val_10
#shuffle the list
random.shuffle(val_ids)

dict_train = {}
dict_val = {}

for img in val_ids:
    if img in reg:
        dict_val[img] = reg[img]

#subtract val_ids entries from the main dict to make train set
dict_train = {k: v for k, v in reg.items() if k not in dict_val}

with open('/scratch/pchaudha/ranking_loss/k/gt_train_rank.json', 'w') as fp:
    json.dump(dict_train, fp, sort_keys = True, indent=4)

with open('/scratch/pchaudha/ranking_loss/k/gt_val_rank.json', 'w') as fp:
    json.dump(dict_val, fp, sort_keys = True, indent=4)

########################################################################################

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

reg_img_id = []
reg_img_label = []

with open('gt_rank_complete.json') as f:
    reg = json.load(f)

for key, value in reg.items():
    # add image path to load it later
    reg_img_id.append(str(key))

    # add image label
    reg_img_label.append(value)
    print(key, value)

print(Counter(reg_img_label))

#make subset for various level values
id_0 = [k for k,v in reg.items() if int(v) == 0]
id_1 = [k for k,v in reg.items() if int(v) == 1]
id_2 = [k for k,v in reg.items() if int(v) == 2]
id_3 = [k for k,v in reg.items() if int(v) == 3]
id_4 = [k for k,v in reg.items() if int(v) == 4]
id_5 = [k for k,v in reg.items() if int(v) == 5]
id_6 = [k for k,v in reg.items() if int(v) == 6]
id_7 = [k for k,v in reg.items() if int(v) == 7]
id_8 = [k for k,v in reg.items() if int(v) == 8]
id_9 = [k for k,v in reg.items() if int(v) == 9]
id_10 = [k for k,v in reg.items() if int(v) == 10]

#shuffle all the lists
random.Random(19).shuffle(id_0)
random.Random(19).shuffle(id_1)
random.Random(19).shuffle(id_2)
random.Random(19).shuffle(id_3)
random.Random(19).shuffle(id_4)
random.Random(19).shuffle(id_5)
random.Random(19).shuffle(id_6)
random.Random(19).shuffle(id_7)
random.Random(19).shuffle(id_8)
random.Random(19).shuffle(id_9)
random.Random(19).shuffle(id_10)

#make folds of equal lengths
[fold0_1 , fold0_2, fold0_3, fold0_4, fold0_5, fold0_6, fold0_7] = chunkIt(id_0,6)
[fold1_1 , fold1_2, fold1_3, fold1_4, fold1_5, fold1_6] = chunkIt(id_1,6)
[fold2_1 , fold2_2, fold2_3, fold2_4, fold2_5, fold2_6] = chunkIt(id_2,6)
[fold3_1 , fold3_2, fold3_3, fold3_4, fold3_5, fold3_6] = chunkIt(id_3,6)
[fold4_1 , fold4_2, fold4_3, fold4_4, fold4_5, fold4_6, fold4_7] = chunkIt(id_4,6)
[fold5_1 , fold5_2, fold5_3, fold5_4, fold5_5, fold5_6, fold5_7] = chunkIt(id_5,6)
[fold6_1 , fold6_2, fold6_3, fold6_4, fold6_5, fold6_6] = chunkIt(id_6,6)
[fold7_1 , fold7_2, fold7_3, fold7_4, fold7_5, fold7_6] = chunkIt(id_7,6)
[fold8_1 , fold8_2, fold8_3, fold8_4, fold8_5, fold8_6, fold8_7] = chunkIt(id_8,6)
[fold9_1 , fold9_2, fold9_3, fold9_4, fold9_5, fold9_6, fold9_7] = chunkIt(id_9,6)
[fold10_1 , fold10_2, fold10_3, fold10_4, fold10_5, fold10_6] = chunkIt(id_10,6)

fold0_6 = fold0_6+fold0_7
fold4_6 = fold4_6+fold4_7
fold5_6 = fold5_6+fold5_7
fold8_6 = fold8_6+fold8_7
fold9_6 = fold9_6+fold9_7

fold1 = fold0_1 + fold1_1 + fold2_1 + fold3_1 + fold4_1 + fold5_1 + fold6_1 + fold7_1 + fold8_1 + fold9_1 + fold10_1
fold2 = fold0_2 + fold1_2 + fold2_2 + fold3_2 + fold4_2 + fold5_2 + fold6_2 + fold7_2 + fold8_2 + fold9_2 + fold10_2
fold3 = fold0_3 + fold1_3 + fold2_3 + fold3_3 + fold4_3 + fold5_3 + fold6_3 + fold7_3 + fold8_3 + fold9_3 + fold10_3
fold4 = fold0_4 + fold1_4 + fold2_4 + fold3_4 + fold4_4 + fold5_4 + fold6_4 + fold7_4 + fold8_4 + fold9_4 + fold10_4
fold5 = fold0_5 + fold1_5 + fold2_5 + fold3_5 + fold4_5 + fold5_5 + fold6_5 + fold7_5 + fold8_5 + fold9_5 + fold10_5
fold6 = fold0_6 + fold1_6 + fold2_6 + fold3_6 + fold4_6 + fold5_6 + fold6_6 + fold7_6 + fold8_6 + fold9_6 + fold10_6

#k=1 files
#train = fold1+fold2+fold3+fold4
#val = fold5
#test = fold6
train1 = fold1 + fold2 + fold3 + fold4
val1 = fold5
test1 = fold6
dict_k_1 = {}
dict_k_1_val = {}
for img in train1:
    if img in reg:
        dict_k_1[img] = reg[img]

for img in val1:
    if img in reg:
        dict_k_1_val[img] = reg[img]

with open('/scratch/pchaudha/ranking_loss/k/k1/gt_train_rank.json', 'w') as fp:
    json.dump(dict_k_1, fp, sort_keys = True, indent=4)

with open('/scratch/pchaudha/ranking_loss/k/k1/gt_val_rank.json', 'w') as fp:
    json.dump(dict_k_1_val, fp, sort_keys = True, indent=4)

#k=2 files
#train = fold5+fold2+fold3+fold4
#val = fold6
#test = fold1
train2 = fold5 + fold2 + fold3 + fold4
val2 = fold6
test2 = fold1
dict_k_2 = {}
dict_k_2_val = {}
for img in train2:
    if img in reg:
        dict_k_2[img] = reg[img]

for img in val2:
    if img in reg:
        dict_k_2_val[img] = reg[img]

with open('/scratch/pchaudha/ranking_loss/k/k2/gt_train_rank.json', 'w') as fp:
    json.dump(dict_k_2, fp, sort_keys = True, indent=4)

with open('/scratch/pchaudha/ranking_loss/k/k2/gt_val_rank.json', 'w') as fp:
    json.dump(dict_k_2_val, fp, sort_keys = True, indent=4)

#k=3 files
#train = fold5+fold6+fold3+fold4
#val = fold1
#test = fold2
train3 = fold5 + fold6 + fold3 + fold4
val3 = fold1
test3 = fold2
dict_k_3 = {}
dict_k_3_val = {}
for img in train3:
    if img in reg:
        dict_k_3[img] = reg[img]

for img in val3:
    if img in reg:
        dict_k_3_val[img] = reg[img]

with open('/scratch/pchaudha/ranking_loss/k/k3/gt_train_rank.json', 'w') as fp:
    json.dump(dict_k_3, fp, sort_keys = True, indent=4)

with open('/scratch/pchaudha/ranking_loss/k/k3/gt_val_rank.json', 'w') as fp:
    json.dump(dict_k_3_val, fp, sort_keys = True, indent=4)

#k=4 files
#train = fold5+fold6+fold1+fold4
#val = fold2
#test = fold3
train4 = fold5 + fold6 + fold1 + fold4
val4 = fold2
test4 = fold3
dict_k_4 = {}
dict_k_4_val = {}
for img in train4:
    if img in reg:
        dict_k_4[img] = reg[img]

for img in val4:
    if img in reg:
        dict_k_4_val[img] = reg[img]

with open('/scratch/pchaudha/ranking_loss/k/k4/gt_train_rank.json', 'w') as fp:
    json.dump(dict_k_4, fp, sort_keys = True, indent=4)

with open('/scratch/pchaudha/ranking_loss/k/k4/gt_val_rank.json', 'w') as fp:
    json.dump(dict_k_4_val, fp, sort_keys = True, indent=4)

#k=5 files
#train = fold5+fold6+fold1+fold2
#val = fold3
#test = fold4
train5 = fold5 + fold6 + fold1 + fold2
val5 = fold3
test5 = fold4
dict_k_5 = {}
dict_k_5_val = {}
for img in train5:
    if img in reg:
        dict_k_5[img] = reg[img]

for img in val5:
    if img in reg:
        dict_k_5_val[img] = reg[img]

with open('/scratch/pchaudha/ranking_loss/k/k5/gt_train_rank.json', 'w') as fp:
    json.dump(dict_k_5, fp, sort_keys = True, indent=4)

with open('/scratch/pchaudha/ranking_loss/k/k5/gt_val_rank.json', 'w') as fp:
    json.dump(dict_k_5_val, fp, sort_keys = True, indent=4)

# dict_fold1 = {}
# dict_fold2 = {}
# dict_fold3 = {}
# dict_fold4 = {}
# dict_fold5 = {}
# dict_fold6 = {}
#
# for img in fold1:
#     if img in reg:
#         dict_fold1[img] = reg[img]
#
# for img in fold2:
#     if img in reg:
#         dict_fold2[img] = reg[img]
#
# for img in fold3:
#     if img in reg:
#         dict_fold3[img] = reg[img]
#
# for img in fold4:
#     if img in reg:
#         dict_fold4[img] = reg[img]
#
# for img in fold5:
#     if img in reg:
#         dict_fold5[img] = reg[img]
#
# for img in fold6:
#     if img in reg:
#         dict_fold6[img] = reg[img]

# with open('gt_reg_fold1.json', 'w') as fp:
#     json.dump(dict_fold1, fp, sort_keys = True, indent=4)
#
# with open('gt_reg_fold2.json', 'w') as fp:
#     json.dump(dict_fold2, fp, sort_keys = True, indent=4)
#
# with open('gt_reg_fold3.json', 'w') as fp:
#     json.dump(dict_fold3, fp, sort_keys = True, indent=4)
#
# with open('gt_reg_fold4.json', 'w') as fp:
#     json.dump(dict_fold4, fp, sort_keys = True, indent=4)
#
# with open('gt_reg_fold5.json', 'w') as fp:
#     json.dump(dict_fold5, fp, sort_keys = True, indent=4)
#
# with open('gt_reg_fold6.json', 'w') as fp:
#     json.dump(dict_fold6, fp, sort_keys = True, indent=4)

print("Done")