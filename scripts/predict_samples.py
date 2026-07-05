"""Precompute qualitative depth predictions + model stats for the notebooks.

Runs each trained model (and DepthAnything) on a handful of NYU test images and
saves, per sample, the RGB / ground-truth depth / valid mask / every model's
predicted depth to a compressed .npz. Also writes meta.json with each model's
trainable + total parameter counts and mean inference time per image.

The notebooks load these artifacts and render the figures, so the notebooks stay
light and run without a GPU (or any checkpoints) at defense time.

    python scripts/predict_samples.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cyclops.data.datasets import build_dataset
from cyclops.models.build import build_model
from cyclops.utils.checkpoint import load_checkpoint
from cyclops.utils.config import load_config
from cyclops.utils.device import resolve_device

CONFIGS = {
    "01_resnet50": "configs/experiments/01_resnet50_nyu.yaml",
    "02_sd_unet": "configs/experiments/02_sd_unet_nyu.yaml",
    "03_ijepa": "configs/experiments/03_ijepa_nyu.yaml",
    "04_fusion": "configs/experiments/04_fusion_nyu.yaml",
    "05_depth_anything": "configs/experiments/05_depth_anything_nyu.yaml",
}
SAMPLES = [0, 120, 300, 540]        # test-set indices to visualize
OUT = Path("outputs/samples")


def main():
    device = resolve_device("auto")
    OUT.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(CONFIGS["01_resnet50"])
    max_depth = base_cfg["dataset"]["depth"]["max"]
    test_set = build_dataset(base_cfg, "test")

    # gather RGB / gt / mask once per sample (same for every model)
    store = {}
    for i in SAMPLES:
        s = test_set[i]
        rgb = np.asarray(Image.open(test_set.rgb_paths[i]).convert("RGB").resize((640, 480)))
        store[i] = {"rgb": rgb, "gt": s["depth"][0].numpy(), "mask": s["mask"][0].numpy(), "preds": {}}

    meta = {}
    for name, path in CONFIGS.items():
        cfg = load_config(path)
        model = build_model(cfg).to(device).eval()
        if cfg["experiment"].get("approach") != 5:
            load_checkpoint(f"outputs/{cfg['experiment']['name']}/checkpoints/best.pt", model)
        align = cfg["eval"].get("align")

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # warmup so the timing excludes lazy CUDA init
        with torch.no_grad():
            model(test_set[SAMPLES[0]]["image"][None].to(device))

        times = []
        for i in SAMPLES:
            image = test_set[i]["image"][None].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                pred = model(image)[0, 0].float()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000.0)

            if align == "median":            # scale zero-shot DepthAnything to the gt median
                gt = torch.from_numpy(store[i]["gt"]).to(device)
                m = torch.from_numpy(store[i]["mask"]).to(device).bool()
                pred = pred * torch.median(gt[m]) / torch.median(pred[m])
            store[i]["preds"][name] = pred.clamp(0, max_depth).cpu().numpy()

        meta[name] = {
            "trainable_params": trainable,
            "total_params": total,
            "inference_ms": float(np.mean(times)),
        }
        print(f"{name:18s} trainable {trainable/1e6:7.2f}M  total {total/1e6:8.2f}M  "
              f"{meta[name]['inference_ms']:.1f} ms/img")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for i in SAMPLES:
        d = store[i]
        np.savez_compressed(
            OUT / f"sample_{i}.npz",
            rgb=d["rgb"], gt=d["gt"], mask=d["mask"],
            **{f"pred_{name}": arr for name, arr in d["preds"].items()},
        )
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {len(SAMPLES)} samples + meta.json to {OUT}")


if __name__ == "__main__":
    main()
