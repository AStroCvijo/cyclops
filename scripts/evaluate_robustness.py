"""Evaluate models on the clean test set and every degraded set (Phase 6/7).

For each experiment config: load the model once, then score it on clean +
fog/blur/exposure x light/moderate/severe (10 conditions), no retraining. Results
are printed as a table and written to outputs/robustness.json.

    python scripts/evaluate_robustness.py                 # all configs/experiments/*.yaml
    python scripts/evaluate_robustness.py --configs configs/experiments/01_resnet50_nyu.yaml ...

Degraded sets must already exist (scripts/generate_degraded.py). A trained config
with no best.pt yet is skipped with a warning; approach 5 needs no checkpoint.
"""

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from cyclops.data.datasets import build_dataset
from cyclops.engine.evaluator import evaluate
from cyclops.models.build import build_model
from cyclops.utils.checkpoint import load_checkpoint
from cyclops.utils.config import load_config
from cyclops.utils.device import dataloader_workers, resolve_device

DEGRADATIONS = ["fog", "blur", "exposure"]
SEVERITIES = ["light", "moderate", "severe"]


# Function for yielding every (label, degradation, severity) test condition
def conditions():
    yield ("clean", None, None)
    for d in DEGRADATIONS:
        for s in SEVERITIES:
            yield (f"{d}/{s}", d, s)


# Function for evaluating one experiment config across all conditions
def evaluate_config(config_path, device):
    cfg = load_config(config_path)
    name = cfg["experiment"]["name"]

    ckpt = f"{cfg['checkpoint']['dir']}/{name}/checkpoints/best.pt"
    if cfg["experiment"].get("approach") != 5 and not Path(ckpt).exists():
        print(f"[skip] {name}: no checkpoint at {ckpt}")
        return None

    model = build_model(cfg).to(device)
    if cfg["experiment"].get("approach") != 5:
        load_checkpoint(ckpt, model)
    align = cfg["eval"].get("align")

    root = cfg["dataset"]["root"]
    results = {}
    for label, degradation, severity in conditions():
        cfg["dataset"]["root"] = (
            root if degradation is None else f"{root}/degraded/{degradation}/{severity}"
        )
        loader = DataLoader(
            build_dataset(cfg, "test"), batch_size=cfg["eval"]["batch_size"], shuffle=False,
            num_workers=dataloader_workers(cfg["training"]["num_workers"]),
        )
        m = evaluate(model, loader, device, cfg, align=align)
        results[label] = m
        print(f"{name:24s} {label:16s} "
              f"abs_rel {m['abs_rel']:.4f}  rmse {m['rmse']:.4f}  delta1 {m['delta1']:.4f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=None,
                    help="experiment configs to evaluate (default: all in configs/experiments)")
    ap.add_argument("--out", default="outputs/robustness.json")
    args = ap.parse_args()

    configs = args.configs or sorted(str(p) for p in Path("configs/experiments").glob("*.yaml"))
    device = resolve_device("auto")

    all_results = {}
    for config_path in configs:
        results = evaluate_config(config_path, device)
        if results is not None:
            all_results[load_config(config_path)["experiment"]["name"]] = results

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
