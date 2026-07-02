"""Scale-invariant log loss (SiLog) — the standard depth training loss.

Compares log(pred) to log(gt) and subtracts a share (`lam`) of the squared
mean error, so a constant scale offset is penalized less than the per-pixel
structure. Computed over valid pixels only. The x10 factor is the usual scaling
that keeps the loss in a convenient range.
"""

import torch
import torch.nn as nn


class SiLogLoss(nn.Module):
    # Function for storing the variance-focus weight lambda
    def __init__(self, lam=0.85, eps=1e-6):
        super().__init__()
        self.lam = lam
        self.eps = eps

    # Function for the scale-invariant log loss over valid pixels
    def forward(self, pred, target, mask):
        mask = mask.bool()
        pred = pred.clamp(min=self.eps)
        g = torch.log(pred[mask]) - torch.log(target[mask])
        return torch.sqrt((g ** 2).mean() - self.lam * g.mean() ** 2 + self.eps) * 10.0
