import torch.nn as nn
import torch

import flag


class PairGeneration(nn.Module):
    def __init__(self, batch_size=18):
        super(PairGeneration, self).__init__()
        self.batch = batch_size
        self.pairs = int((batch_size*(batch_size-1))/2)

    def forward(self, x):
        #x = x.squeeze().tolist()
        count = 0
        x1 = torch.cuda.FloatTensor(self.pairs).fill_(0)
        x2 = torch.cuda.FloatTensor(self.pairs).fill_(0)
        #print("output for ranking: ", x)
        for i in range(0, self.batch):
            for j in range(i+1, self.batch):
                #print("pair: ", count)
                x1[count] = x[i]
                x2[count] = x[j]
                count = count + 1

        return x1,x2

class PreeNet(nn.Module):
    def __init__(self,model_vgg):
        super(PreeNet, self).__init__()
        self.vgg = model_vgg
        #self.avg_pool = nn.AvgPool2d()
        #self.conv_new = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        #self.fc1 = nn.Linear(out_features=1)
        self.pair = PairGeneration()

    def forward(self, x):

        x1 = self.vgg(x)

        #regression head
        #out = F.relu(self.conv_new(x1))
        #reg = self.fc1(out)

        #rank head
        #rank = F.avg_pool2d(out)
        #rank = self.pair(rank)

        if flag.TRAIN_RANK == True:
            model = self.pair(x1)
            return model
        else:
            return x1