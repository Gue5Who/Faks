import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn


# og koda iz spletne
#class ModelCT(nn.Module):
#    def __init__(self):
#        super(ModelCT, self).__init__()
#        self.backbone =resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
#        self.backbone.conv1 = nn.Conv2d(1, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias=False)
#        self.convolution2d = nn.Conv2d(512, 1,kernel_size=(1, 1), stride=(1, 1), bias=True)
#        self.fc_maxpool = nn.AdaptiveMaxPool2d((1, 1))
#
#    def forward(self, x):
#        x = self.backbone.conv1(x)
#        x = self.backbone.bn1(x)
#        x = self.backbone.relu(x)
#        x = self.backbone.maxpool(x)
#        x = self.backbone.layer1(x)
#        x = self.backbone.layer2(x)
#        x = self.backbone.layer3(x)
#        x = self.backbone.layer4(x)
#        x = self.convolution2d(x)
#        x = self.fc_maxpool(x)
#        x = torch.flatten(x, 1)
#
#        return x


class ModelCT(nn.Module):
    def __init__(self):
        super(ModelCT, self).__init__()

        # Vhodne slike imajo obliko (batch_size, 1, 512, 512)
        in_features = 512 * 512

        self.fc1 = nn.Linear(in_features, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 1)   # en izhodni logit (za BCEWithLogitsLoss)

    def forward(self, x):
        # x: (batch_size, 1, 512, 512)
        x = torch.flatten(x, 1)  # -> (batch_size, 262144)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)          # brez sigmoid, ker to dela BCEWithLogitsLoss

        return x
