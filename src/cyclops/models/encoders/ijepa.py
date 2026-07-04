"""Frozen I-JEPA ViT encoder: returns patch-token feature maps for the decoder.

I-JEPA is a self-supervised ViT (learns by predicting the latent representation
of masked regions). We run it frozen, tap a few transformer layers, and reshape
each layer's patch tokens back to a 2D grid. Every tap comes out at the same
(coarse) patch resolution; the DPT decoder turns these grids into a depth map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import IJepaModel

from cyclops.models.encoders.base import Encoder


class IjepaEncoder(Encoder):
    # Function for loading a frozen I-JEPA ViT and picking which layers to tap
    def __init__(self, model_name="facebook/ijepa_vith14_1k",
                 tap_layers=(8, 16, 24, 32), image_size=(420, 560)):
        super().__init__()
        self.vit = IJepaModel.from_pretrained(model_name)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

        self.tap_layers = list(tap_layers)
        self.image_size = tuple(image_size)             # ViT input, divisible by patch
        self.patch = self.vit.config.patch_size
        hidden = self.vit.config.hidden_size
        self.out_channels = [hidden, hidden, hidden, hidden]   # all taps are ViT hidden dim

    # Function for keeping the frozen ViT in eval mode even when the model trains
    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()
        return self

    # Function for reshaping a token sequence [B, N, C] to a grid [B, C, H, W]
    def _tokens_to_grid(self, tokens, gh, gw):
        b, n, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, gh, gw)

    # Function for returning the tapped token grids (all at the ViT patch grid)
    def forward(self, x):
        x = F.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False)
        gh = self.image_size[0] // self.patch
        gw = self.image_size[1] // self.patch
        with torch.no_grad():
            out = self.vit(x, output_hidden_states=True, interpolate_pos_encoding=True)
        return [self._tokens_to_grid(out.hidden_states[i], gh, gw) for i in self.tap_layers]
