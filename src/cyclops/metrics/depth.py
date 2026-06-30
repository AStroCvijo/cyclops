import torch


# Function for keeping only the pixels we should score (gt depth > 0, and inside mask if given)
def keep_valid(pred, target, mask=None):
    valid = target > 0
    if mask is not None:
        valid = valid & mask.bool()
    return pred[valid], target[valid]


# Function for the mean absolute relative error: mean(|pred - target| / target)
def abs_rel(pred, target):
    return (torch.abs(pred - target) / target).mean().item()


# Function for the root mean squared error, in meters
def rmse(pred, target):
    return torch.sqrt(((pred - target) ** 2).mean()).item()


# Function for the fraction of pixels where max(pred/target, target/pred) < 1.25
def delta1(pred, target):
    ratio = torch.maximum(pred / target, target / pred)
    return (ratio < 1.25).float().mean().item()


# Function for computing all three metrics at once and returning them as a dict
def compute_metrics(pred, target, mask=None):
    pred, target = keep_valid(pred, target, mask)
    return {
        "abs_rel": abs_rel(pred, target),
        "rmse": rmse(pred, target),
        "delta1": delta1(pred, target),
    }
