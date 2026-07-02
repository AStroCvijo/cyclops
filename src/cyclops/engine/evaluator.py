"""Evaluate a depth model on a loader and return averaged metrics.

Predictions are clamped to the dataset depth range, and for NYU the standard
Eigen center crop is applied before scoring (so numbers are comparable to the
literature). `align="median"` rescales each prediction to the gt median — used
only for relative-depth SOTA models (approach 5), not for the trained metric models.
"""

import torch

from cyclops.metrics.depth import compute_metrics


# Function for the Eigen center-crop mask used for NYU evaluation
def eigen_crop_mask(h, w, device):
    mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    # standard NYU Eigen crop, expressed as fractions so it scales with size
    top, bottom = int(0.09375 * h), int(0.98125 * h)      # ~45..471 at h=480
    left, right = int(0.03594 * w), int(0.96406 * w)      # ~23..617 at w=640
    mask[top:bottom, left:right] = True
    return mask


# Function for evaluating a model on a loader and returning averaged metrics
@torch.no_grad()
def evaluate(model, loader, device, cfg, align=None):
    model.eval()
    dmin = cfg["dataset"]["depth"]["min"]
    dmax = cfg["dataset"]["depth"]["max"]
    crop = cfg["dataset"].get("eval_crop")

    totals = {"abs_rel": 0.0, "rmse": 0.0, "delta1": 0.0}
    batches = 0
    for batch in loader:
        image = batch["image"].to(device)
        depth = batch["depth"].to(device)
        mask = batch["mask"].to(device)

        pred = model(image).clamp(dmin, dmax)

        if crop == "eigen":
            h, w = depth.shape[-2:]
            mask = mask & eigen_crop_mask(h, w, device)[None, None]

        if align == "median":
            for b in range(pred.shape[0]):
                mb = mask[b]
                if mb.any():
                    pred[b] = pred[b] * torch.median(depth[b][mb]) / torch.median(pred[b][mb])

        m = compute_metrics(pred, depth, mask)
        for k in totals:
            totals[k] += m[k]
        batches += 1

    return {k: totals[k] / batches for k in totals}
