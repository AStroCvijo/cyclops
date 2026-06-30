import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from cyclops.data.datasets import build_dataset
from cyclops.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from cyclops.utils.config import load_config


# Function for undoing the ImageNet normalization so the image is viewable
def denorm(image_tensor):
    image = image_tensor.permute(1, 2, 0).numpy()
    return np.clip(image * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="outputs/data_check.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset = build_dataset(cfg, args.split)
    print(f"{args.split}: {len(dataset)} samples")

    loader = DataLoader(dataset, batch_size=args.n, shuffle=True)
    batch = next(iter(loader))
    images, depths, masks = batch["image"], batch["depth"], batch["mask"]

    # depth range over valid pixels — NYU should land around 0..10 meters
    valid = depths[masks]
    print(f"valid depth: min={valid.min():.2f} max={valid.max():.2f} mean={valid.mean():.2f} m")
    print(f"image tensor: shape={tuple(images.shape)}  depth: shape={tuple(depths.shape)}")

    fig, axes = plt.subplots(2, args.n, figsize=(4 * args.n, 6))
    for j in range(args.n):
        axes[0, j].imshow(denorm(images[j]))
        axes[0, j].set_title("rgb")
        axes[0, j].axis("off")
        axes[1, j].imshow(depths[j, 0].numpy(), cmap="magma")
        axes[1, j].set_title("depth (m)")
        axes[1, j].axis("off")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print("saved", args.out)


if __name__ == "__main__":
    main()
