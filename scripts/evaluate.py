"""Evaluate a trained checkpoint on the test split.

    python scripts/evaluate.py --config configs/experiments/01_resnet50_nyu.yaml
    python scripts/evaluate.py --config ... --checkpoint outputs/01_resnet50_nyu/checkpoints/best.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader

from cyclops.data.datasets import build_dataset
from cyclops.engine.evaluator import evaluate
from cyclops.models.build import build_model
from cyclops.utils.checkpoint import load_checkpoint
from cyclops.utils.config import load_config
from cyclops.utils.device import dataloader_workers, resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None, help="defaults to the experiment's best.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "auto"))

    model = build_model(cfg).to(device)
    ckpt = args.checkpoint or (
        f"{cfg['checkpoint']['dir']}/{cfg['experiment']['name']}/checkpoints/best.pt"
    )
    load_checkpoint(ckpt, model)

    test_set = build_dataset(cfg, "test")
    loader = DataLoader(
        test_set, batch_size=cfg["eval"]["batch_size"], shuffle=False,
        num_workers=dataloader_workers(cfg["training"]["num_workers"]),
    )
    metrics = evaluate(model, loader, device, cfg)
    print(metrics)


if __name__ == "__main__":
    main()
