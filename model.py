import torch 
import torch.nn as nn 
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # define your layer ops.
        self.conv1 = nn.Conv2d(3, 6, 7, stride=2)
        self.norm1= nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=2)
        self.norm2= nn.BatchNorm2d(16)
        self.conv3= nn.Conv2d(16, 32, 5, stride=2)
        self.norm3= nn.BatchNorm2d(32)
        self.conv4= nn.Conv2d(32, 64, 3, stride=2)
        self.norm4= nn.BatchNorm2d(64)
        self.conv5= nn.Conv2d(64, 64, 3, stride=2)
        self.norm5= nn.BatchNorm2d(64)
        self.flatten= Flatten()
        self.fc1 = nn.Linear(2304, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.1)
        self.fc_norm1 = nn.BatchNorm1d(64)
        self.fc_norm2 = nn.BatchNorm1d(64)


    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = F.relu(self.norm4(self.conv4(x)))
        x = F.relu(self.norm5(self.conv5(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x).squeeze()
        return x

    

