"""ResNet-50 encoder: returns the 4 stage feature maps for the decoder.

Feature maps come out fine -> coarse at strides 4, 8, 16, 32 with channels
256, 512, 1024, 2048.
"""

import torch.nn as nn
import torchvision

from cyclops.models.encoders.base import Encoder


class ResNet50Encoder(Encoder):
    out_channels = [256, 512, 1024, 2048]   # at strides 4, 8, 16, 32

    # Function for setting up a (pretrained) ResNet-50 backbone
    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = torchvision.models.resnet50(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)   # -> /4
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    # Function for returning the 4 feature maps (fine -> coarse)
    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)    # /4,  256
        f2 = self.layer2(f1)   # /8,  512
        f3 = self.layer3(f2)   # /16, 1024
        f4 = self.layer4(f3)   # /32, 2048
        return [f1, f2, f3, f4]
