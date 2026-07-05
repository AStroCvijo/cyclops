"""Pick the best available torch device for the current machine.

Config `device` values:
  auto  — cuda if available, else Apple MPS, else cpu (works everywhere)
  cuda  — NVIDIA GPU; falls back to mps/cpu with a warning if missing
  mps   — Apple Silicon GPU; falls back to cpu with a warning if missing
  cpu   — force CPU
"""

import os

import torch

from cyclops.utils.logging import get_logger

log = get_logger(__name__)


def resolve_device(requested: str = "auto") -> torch.device:
    requested = (requested or "auto").lower()

    if requested == "cpu":
        return torch.device("cpu")

    if requested in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")

    if requested in ("mps", "auto") and _mps_available():
        return torch.device("mps")

    if requested == "cuda":
        log.warning("cuda requested but unavailable — falling back to cpu")
    elif requested == "mps":
        log.warning("mps requested but unavailable — falling back to cpu")

    return torch.device("cpu")


def pin_memory_for(device: torch.device) -> bool:
    return device.type == "cuda"


def dataloader_workers(requested: int) -> int:
    # Windows uses spawn; multiprocess loaders often hang or crash.
    if os.name == "nt" and requested > 0:
        log.warning("num_workers=%d on Windows — using 0 instead", requested)
        return 0
    return requested


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
