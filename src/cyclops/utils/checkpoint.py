from pathlib import Path

import torch


# Function for saving the model (and optionally optimizer) state to `path`
def save_checkpoint(path, model, optimizer=None, epoch=0, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {"model": model.state_dict(), "epoch": epoch, "metrics": metrics or {}}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


# Function for loading weights from `path` into `model` (and optimizer if given)
def load_checkpoint(path, model, optimizer=None):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state
