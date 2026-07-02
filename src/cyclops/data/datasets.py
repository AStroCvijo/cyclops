from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cyclops.data.transforms import augment, to_tensor


class NYUDataset(Dataset):
    # Function for collecting all RGB paths and remembering the dataset settings
    def __init__(self, root, image_size, depth_min, depth_max, depth_scale=1000.0, train=True):
        self.root = Path(root)
        self.image_size = tuple(image_size)   # (H, W)
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_scale = depth_scale
        self.train = train

        self.rgb_paths = sorted(self.root.rglob("rgb_*.jpg"))
        if not self.rgb_paths:
            raise FileNotFoundError(f"No rgb_*.jpg images found under {self.root}")

    # Function for the number of samples
    def __len__(self):
        return len(self.rgb_paths)

    # Function for the depth path that matches an RGB path (same folder, renamed)
    def _depth_path(self, rgb_path):
        name = rgb_path.name.replace("rgb_", "sync_depth_").replace(".jpg", ".png")
        return rgb_path.with_name(name)

    # Function for loading one sample as {image, depth, mask}
    def __getitem__(self, i):
        rgb_path = self.rgb_paths[i]
        image = Image.open(rgb_path).convert("RGB")
        depth = Image.open(self._depth_path(rgb_path))

        # resize: bilinear for RGB, nearest for depth (don't blur depth edges)
        H, W = self.image_size
        image = image.resize((W, H), Image.BILINEAR)
        depth = depth.resize((W, H), Image.NEAREST)

        image = np.asarray(image, dtype=np.float32) / 255.0
        depth = np.asarray(depth, dtype=np.float32) / self.depth_scale   # -> meters

        if self.train:
            image, depth = augment(image, depth)

        # valid = real measurement within the sensor range (zeros are missing depth)
        mask = (depth > self.depth_min) & (depth < self.depth_max)
        depth = np.clip(depth, 0.0, self.depth_max)

        return {
            "image": to_tensor(image),                # (3,H,W) normalized
            "depth": torch.from_numpy(depth)[None],   # (1,H,W) meters
            "mask": torch.from_numpy(mask)[None],     # (1,H,W) bool
        }


# Function for building the dataset for a split ("train"/"test") from the config
def build_dataset(cfg, split):
    ds = cfg["dataset"]
    if ds["name"] != "nyu_depth_v2":
        raise ValueError(f"only NYU is implemented so far, got {ds['name']!r}")
    root = Path(ds["root"]).expanduser() / ds["splits"][split]   # root + relative split
    return NYUDataset(
        root=root,
        image_size=ds["image_size"],
        depth_min=ds["depth"]["min"],
        depth_max=ds["depth"]["max"],
        depth_scale=ds["depth"].get("scale", 1000.0),
        train=(split == "train"),
    )
