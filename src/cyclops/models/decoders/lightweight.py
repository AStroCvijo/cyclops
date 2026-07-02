"""Lightweight top-down decoder: feature pyramid -> dense depth map.

Starts from the coarsest encoder feature, upsamples step by step, and at each
step concatenates the matching finer feature (skip connection). The final head
produces one channel, scaled through a sigmoid to (0, max_depth) meters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Function for a 3x3 conv + BatchNorm + ReLU block
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class LightweightDecoder(nn.Module):
    # Function for building the decoder from encoder channels and decoder widths
    def __init__(self, in_channels, widths, max_depth):
        super().__init__()
        self.max_depth = max_depth
        c1, c2, c3, c4 = in_channels     # fine -> coarse: 256, 512, 1024, 2048
        d0, d1, d2, d3 = widths          # 256, 128, 64, 32

        self.reduce = conv_block(c4, d0)         # coarsest feature -> d0
        self.up1 = conv_block(d0 + c3, d1)
        self.up2 = conv_block(d1 + c2, d2)
        self.up3 = conv_block(d2 + c1, d3)
        self.head = nn.Conv2d(d3, 1, 1)

    # Function for upsampling x to a skip's size and concatenating them
    def _up_cat(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    # Function for predicting a depth map from the encoder features
    def forward(self, feats, out_hw):
        f1, f2, f3, f4 = feats
        x = self.reduce(f4)                  # /32
        x = self.up1(self._up_cat(x, f3))    # /16
        x = self.up2(self._up_cat(x, f2))    # /8
        x = self.up3(self._up_cat(x, f1))    # /4
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.head(x)) * self.max_depth   # (0, max_depth) meters
