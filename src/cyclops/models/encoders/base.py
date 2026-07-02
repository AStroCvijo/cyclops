"""The encoder interface every backbone follows.

A subclass sets `out_channels` (the channel count of each feature map it
returns, ordered fine -> coarse) and implements forward(image) -> list of
feature maps in that same order. The decoder relies only on this contract, so
swapping ResNet-50 for another encoder needs no decoder changes.
"""

import torch.nn as nn


class Encoder(nn.Module):
    out_channels = []
