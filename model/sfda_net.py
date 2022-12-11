import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
import torchvision


class SFDANet(nn.Module):
    def __init__(self, num_features_in=256, feature_size=256):
        super(SFDANet, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(num_features_in, num_features_in, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        return out

