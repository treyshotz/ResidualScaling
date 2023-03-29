from typing import Any

from torchvision.models import resnet50
from torch import Tensor

from torchvision.models.resnet import Bottleneck, _resnet, ResNet


class ScaledBottleneck(Bottleneck):

    def __init__(self, scaling_factor: int, inplanes: int, planes: int):
        super().__init__(inplanes, planes)
        self.scaling_factor = scaling_factor

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += (identity * self.scaling_factor)
        out = self.relu(out)

        return out


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet("resnet50", ScaledBottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
