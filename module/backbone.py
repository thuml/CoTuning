import torch.nn as nn

from torchvision import models

__all__ = ['ResNet50_F', 'ResNet50_C']


class ResNet50_F(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50_F, self).__init__()
        model_resnet50 = models.resnet50(pretrained=pretrained)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    @property
    def output_dim(self):
        return self.__in_features


class ResNet50_C(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50_C, self).__init__()
        model_resnet50 = models.resnet50(pretrained=pretrained)
        self.fc = model_resnet50.fc

    def forward(self, x):
        x = self.fc(x)

        return x
