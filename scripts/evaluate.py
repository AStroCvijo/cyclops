"""Evaluate a trained checkpoint on the test split (clean or degraded).

    python scripts/evaluate.py --config configs/experiments/01_resnet50_nyu.yaml
    python scripts/evaluate.py --config ... --degradation fog --severity severe

`--degradation` repoints the test set at a set written by scripts/generate_degraded.py.
Approach 5 (DepthAnything) is zero-shot, so no checkpoint is loaded for it.
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
    ap.add_argument("--degradation", default=None, choices=["fog", "blur", "exposure"])
    ap.add_argument("--severity", default=None, choices=["light", "moderate", "severe"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.get("device", "auto"))

    # point the test set at a degraded copy (mirrors the clean layout under degraded/)
    if args.degradation:
        if not args.severity:
            ap.error("--severity is required with --degradation")
        cfg["dataset"]["root"] = (
            f"{cfg['dataset']['root']}/degraded/{args.degradation}/{args.severity}"
        )

    model = build_model(cfg).to(device)
    if cfg["experiment"].get("approach") != 5:            # approach 5 is zero-shot, no checkpoint
        ckpt = args.checkpoint or (
            f"{cfg['checkpoint']['dir']}/{cfg['experiment']['name']}/checkpoints/best.pt"
        )
        load_checkpoint(ckpt, model)

    test_set = build_dataset(cfg, "test")
    loader = DataLoader(
        test_set, batch_size=cfg["eval"]["batch_size"], shuffle=False,
        num_workers=dataloader_workers(cfg["training"]["num_workers"]),
    )
    metrics = evaluate(model, loader, device, cfg, align=cfg["eval"].get("align"))
    print(metrics)


if __name__ == "__main__":
    main()
