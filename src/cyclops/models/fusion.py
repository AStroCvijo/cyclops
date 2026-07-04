"""Fusion of two frozen encoders (approach 4).

SD-UNet features are texture-rich and geometrically precise; I-JEPA features are
semantic and structural. We run both encoders frozen and fuse their feature maps
level by level, then a trainable decoder regresses depth. Two fusion methods:

  - `concat`: parameter-free channel concatenation (the simple baseline).
  - `cross_attention`: at each level the first encoder's features (queries) attend
    to the second encoder's features (keys/values).

The first encoder in the list sets the spatial size of each level; the other
encoder's maps are resized to match. The fusion module (if it has parameters) and
the decoder train; the encoders stay frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Function for resizing a feature map to a target spatial size
def resize_to(x, size):
    if x.shape[-2:] == size:
        return x
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ConcatFusion(nn.Module):
    # Function for concatenating each pyramid level of two encoders (no parameters)
    def __init__(self, channels_a, channels_b):
        super().__init__()
        self.out_channels = [ca + cb for ca, cb in zip(channels_a, channels_b)]

    # Function for fusing two feature pyramids level by level (encoder a sets the size)
    def forward(self, feats_a, feats_b):
        fused = []
        for fa, fb in zip(feats_a, feats_b):
            fb = resize_to(fb, fa.shape[-2:])
            fused.append(torch.cat([fa, fb], dim=1))
        return fused


class CrossAttentionFusion(nn.Module):
    # Function for building per-level projections and cross-attention (trainable)
    def __init__(self, channels_a, channels_b, dim=256, heads=8):
        super().__init__()
        self.out_channels = [dim] * len(channels_a)
        self.to_q = nn.ModuleList([nn.Conv2d(ca, dim, 1) for ca in channels_a])
        self.to_k = nn.ModuleList([nn.Conv2d(cb, dim, 1) for cb in channels_b])
        self.to_v = nn.ModuleList([nn.Conv2d(cb, dim, 1) for cb in channels_b])
        self.attn = nn.ModuleList(
            [nn.MultiheadAttention(dim, heads, batch_first=True) for _ in channels_a]
        )

    # Function for fusing each level: encoder-a tokens attend to encoder-b tokens
    def forward(self, feats_a, feats_b):
        fused = []
        for i, (fa, fb) in enumerate(zip(feats_a, feats_b)):
            b, _, h, w = fa.shape
            q = self.to_q[i](fa).flatten(2).transpose(1, 2)      # (B, HcWc, dim)
            k = self.to_k[i](fb).flatten(2).transpose(1, 2)
            v = self.to_v[i](fb).flatten(2).transpose(1, 2)
            out, _ = self.attn[i](q, k, v)                       # (B, HcWc, dim)
            fused.append(out.transpose(1, 2).reshape(b, -1, h, w))
        return fused
