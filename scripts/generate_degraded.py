"""Generate degraded test sets to disk (Phase 6, robustness).

Applies homogeneous fog / Gaussian blur / under-exposure to the NYU test RGBs at
three severities each (light, moderate, severe) and writes them next to a copy of
the *unchanged* depth PNGs. Every model is later evaluated on these fixed images
without retraining, so the degradation must be written once and be identical for
all models — hence deterministic (fixed seed) and generated to disk.

CPU-only: no model and no GPU are involved, just Albumentations image ops.

    python scripts/generate_degraded.py --degradation fog
    python scripts/generate_degraded.py --degradation all --severity all

Output layout (mirrors the source test dir so only the dataset root changes at eval):

    data/nyu_depth_v2/degraded/<name>/<severity>/rgb_*.jpg   (degraded)
                                                /sync_depth_*.png (copied, unchanged)
"""

import argparse
import random
import shutil
from pathlib import Path

import albumentations as A
import numpy as np
import yaml
from PIL import Image

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SEVERITIES = ["light", "moderate", "severe"]

# The --degradation flag names a config file in configs/degradations/.
DEGRADATIONS = {
    "fog": "fog.yaml",
    "blur": "blur.yaml",
    "exposure": "exposure.yaml",
}


# Function for building the deterministic Albumentations transform for one severity
def build_transform(name, params):
    if name == "fog":
        # homogeneous fog: fog_coef sets density, alpha_coef the haze transparency
        c = params["fog_coef"]
        return A.RandomFog(fog_coef_range=(c, c), alpha_coef=params["alpha_coef"], p=1.0)
    if name == "gaussian_blur":
        k = params["kernel_size"]
        s = params["sigma"]
        return A.GaussianBlur(blur_limit=(k, k), sigma_limit=(s, s), p=1.0)
    if name == "exposure":
        # darken (brightness < 0) then crush shadows (gamma > 1); gamma is percent here
        g = int(params["gamma"] * 100)
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(params["brightness"], params["brightness"]),
                contrast_limit=0.0, p=1.0,
            ),
            A.RandomGamma(gamma_limit=(g, g), p=1.0),
        ])
    raise ValueError(f"unknown degradation {name!r}")


# Function for locating the source test dir from the dataset config
def test_dir(dataset_cfg):
    ds = yaml.safe_load(Path(dataset_cfg).read_text())["dataset"]
    return Path(ds["root"]).expanduser() / ds["splits"]["test"], Path(ds["root"]).expanduser()


# Function for degrading every RGB in the test dir at one severity
def generate(name, severity, transform, source, out_dir):
    rgb_paths = sorted(source.rglob("rgb_*.jpg"))
    print(f"[{name}/{severity}] {len(rgb_paths)} images -> {out_dir}")
    for i, rgb_path in enumerate(rgb_paths):
        rel = rgb_path.relative_to(source)
        out_rgb = out_dir / rel
        out_rgb.parent.mkdir(parents=True, exist_ok=True)

        # deterministic per image: same seed -> same fog particles every run/machine
        random.seed(i)
        np.random.seed(i)
        image = np.asarray(Image.open(rgb_path).convert("RGB"))
        degraded = transform(image=image)["image"]
        Image.fromarray(degraded).save(out_rgb, quality=95)

        # depth is geometry, not appearance — copy it unchanged next to the RGB
        depth_name = rgb_path.name.replace("rgb_", "sync_depth_").replace(".jpg", ".png")
        shutil.copy2(rgb_path.with_name(depth_name), out_rgb.with_name(depth_name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degradation", default="all", choices=[*DEGRADATIONS, "all"])
    ap.add_argument("--severity", default="all", choices=[*SEVERITIES, "all"])
    ap.add_argument("--dataset-config", default=str(CONFIG_DIR / "_base" / "nyu.yaml"))
    args = ap.parse_args()

    source, root = test_dir(args.dataset_config)
    if not source.exists():
        raise FileNotFoundError(f"test dir not found: {source}")

    names = list(DEGRADATIONS) if args.degradation == "all" else [args.degradation]
    levels = SEVERITIES if args.severity == "all" else [args.severity]

    for name in names:
        cfg = yaml.safe_load((CONFIG_DIR / "degradations" / DEGRADATIONS[name]).read_text())
        deg = cfg["degradation"]
        for severity in levels:
            transform = build_transform(deg["name"], deg["severities"][severity])
            out_dir = root / "degraded" / name / severity / source.relative_to(root)
            generate(name, severity, transform, source, out_dir)


if __name__ == "__main__":
    main()
