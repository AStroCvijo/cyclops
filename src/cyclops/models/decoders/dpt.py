"""DPT-style decoder: reassemble ViT token grids into a dense depth map.

The I-JEPA encoder hands us several token grids, all at the same coarse patch
resolution. We reassemble them into a feature pyramid (fine -> coarse) with 1x1
projections and resampling, then fuse top-down into a depth map exactly like the
lightweight decoder. This whole module is trainable; the encoder stays frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cyclops.models.decoders.lightweight import conv_block


class DPTDecoder(nn.Module):
    # Function for building the reassembly projections and the top-down fusion
    def __init__(self, in_channels, reassemble_channels, widths, max_depth):
        super().__init__()
        self.max_depth = max_depth
        self.scales = [4.0, 2.0, 1.0, 0.5]        # fine -> coarse resampling of the patch grid

        self.project = nn.ModuleList([
            nn.Conv2d(cin, rc, 1) for cin, rc in zip(in_channels, reassemble_channels)
        ])

        c1, c2, c3, c4 = reassemble_channels      # fine -> coarse feature channels
        d0, d1, d2, d3 = widths
        self.reduce = conv_block(c4, d0)          # coarsest feature -> d0
        self.up1 = conv_block(d0 + c3, d1)
        self.up2 = conv_block(d1 + c2, d2)
        self.up3 = conv_block(d2 + c1, d3)
        self.head = nn.Conv2d(d3, 1, 1)

    # Function for projecting and resampling each token grid to its pyramid level
    def _reassemble(self, feats):
        out = []
        for f, proj, scale in zip(feats, self.project, self.scales):
            x = proj(f)
            x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
            out.append(x)
        return out

    # Function for upsampling x to a skip's size and concatenating them
    def _up_cat(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    # Function for predicting a depth map from the encoder token grids
    def forward(self, feats, out_hw):
        f1, f2, f3, f4 = self._reassemble(feats)
        x = self.reduce(f4)                  # coarsest
        x = self.up1(self._up_cat(x, f3))
        x = self.up2(self._up_cat(x, f2))
        x = self.up3(self._up_cat(x, f1))
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.head(x)) * self.max_depth   # (0, max_depth) meters
