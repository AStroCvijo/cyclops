"""DepthAnything V2 zero-shot depth (approach 5, eval-only — the SOTA upper bound).

Not part of the encoder/decoder pipeline: a full pretrained model that predicts
depth directly, used without any training. We use the *metric-indoor* checkpoint,
which regresses depth in meters (trained on Hypersim), so its output is directly
comparable to NYU ground truth — no scale/shift alignment needed. (The relative
checkpoints output affine-invariant inverse depth, which median scaling can't fully
correct, so they score far worse than they should on a metric benchmark.)

The DINOv2 backbone uses ImageNet normalization, the same as `data/transforms.py`,
so the already-normalized batch is fed straight in — only the spatial size is
adjusted to a multiple of 14 (the ViT patch size).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation


class DepthAnythingModel(nn.Module):
    # Function for loading a frozen DepthAnything model for zero-shot depth
    def __init__(self, model_id):
        super().__init__()
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    # Function for keeping the backbone in eval mode even under model.train()
    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    # Function for predicting a metric depth map (meters)
    @torch.no_grad()
    def forward(self, image):
        H, W = image.shape[-2:]
        h = max(14, round(H / 14) * 14)      # DepthAnything needs sizes divisible by 14
        w = max(14, round(W / 14) * 14)
        x = F.interpolate(image, size=(h, w), mode="bilinear", align_corners=False)

        depth = self.model(pixel_values=x).predicted_depth   # (B, h, w), meters
        return F.interpolate(depth[:, None], size=(H, W), mode="bilinear", align_corners=False)
