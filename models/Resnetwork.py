from typing import Any

import torch
from torch import nn
from torchvision.models.resnet import Bottleneck, _resnet, ResNet

from models.ScaledBottleNeck import ScaledBottleneck


def custom_resnet50(progress: bool = True, scaled=True, **kwargs: Any) -> ResNet:
    if scaled:
        return _resnet(ScaledBottleneck, [3, 4, 6, 3], weights=None, progress=False, **kwargs)
    else:
        return _resnet(Bottleneck, [3, 4, 6, 3], weights=None, progress=False, **kwargs)


class ResNetNetwork(torch.nn.Module):

    def __init__(self, scaled=True):
        super().__init__()

        self.net = custom_resnet50(scaled=scaled)
        self.fc_in_features = self.net.fc.in_features

        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(1, 1))
        self.net = nn.Sequential(*(list(self.net.children())[:-1]))

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10), )

    def forward(self, x):
        # Forward pass
        output = self.net(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output